#!/usr/bin/env python3
"""Run the current OCR pipeline on val / gold_test and write metrics + error CSV.

Primary metric: mean sentence-level similarity (see ``eval/metrics.py``).
``gold_test`` is scored for reporting only — never used for tuning here.

Production OCR backend is the VLM path in ``pipeline/ocr.py`` (OpenAI-compatible
vision API). Local Surya recognition is the free zero-shot baseline.

Usage:
    python -m eval.split                          # document split → eval/splits_by_document
    python -m eval.run_baseline                   # line_crops + qwen_local (defaults)
    python -m eval.run_baseline --splits val --error-analysis
    QWEN_LORA_PATH=models/qwen3-vl-fa-exam-lora-v1 python -m eval.run_baseline --splits val
    python -m eval.run_baseline --mode page --ocr-backend surya --diagnose-matching
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

from eval.error_analysis import analyze_units, write_worst_csv  # noqa: E402
from eval.metrics import (  # noqa: E402
    aggregate_results,
    evaluate_pair,
    write_error_csv,
)
from eval.normalize import normalize_text  # noqa: E402
from eval.split import (  # noqa: E402
    DEFAULT_OUT_DIR,
    audit_gold,
    build_split_manifests,
    load_jsonl,
    save_manifests,
)

DEFAULT_GOLD = PROJECT_ROOT / "dataset" / "gold.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "eval"

OCR_BACKENDS = ("surya", "vlm", "qwen_local", "empty")

_DEFAULT_QWEN_MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
_DEFAULT_QWEN_PRED_CACHE = PROJECT_ROOT / "dataset" / "pred_qwen3-vl-2b-persian-arabic-ocr-v1.0.jsonl"


def _ensure_splits(split_dir: Path, gold_path: Path, seed: int) -> dict[str, dict]:
    needed = [split_dir / f"{s}.json" for s in ("train", "val", "gold_test")]
    if all(p.exists() for p in needed):
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in needed}
    # Infer split mode from directory name; default to document (Phase C).
    split_by = "page" if split_dir.name == "splits" else "document"
    print(
        f"[baseline] splits missing under {split_dir}; "
        f"creating with seed={seed} split_by={split_by}"
    )
    manifests = build_split_manifests(
        load_jsonl(gold_path), seed=seed, split_by=split_by
    )
    save_manifests(manifests, split_dir)
    return manifests


def _page_reference(page_item: dict) -> str:
    lines = []
    for row in page_item.get("lines", []):
        text = (row.get("text") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _ocr_crops_surya(crops: list[Image.Image], *, batch_size: int = 16) -> list[str]:
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from surya.recognition import batch_recognition

    if not crops:
        return []
    if not hasattr(_ocr_crops_surya, "_model"):
        _ocr_crops_surya._model = load_rec_model()  # type: ignore[attr-defined]
        _ocr_crops_surya._processor = load_rec_processor()  # type: ignore[attr-defined]
    texts: list[str] = []
    for start in range(0, len(crops), batch_size):
        chunk = crops[start : start + batch_size]
        langs = [["fa"]] * len(chunk)
        preds = batch_recognition(
            chunk,
            langs,
            _ocr_crops_surya._model,  # type: ignore[attr-defined]
            _ocr_crops_surya._processor,  # type: ignore[attr-defined]
        )
        part = preds[0] if isinstance(preds, tuple) else preds
        texts.extend(str(t or "") for t in part)
    return texts


def _ocr_crop_vlm(crop: Image.Image) -> str:
    """Production VLM path: same prep as ``pipeline.ocr.run_ocr``."""
    from pipeline.ocr import _pil_to_base64, _prepare_ocr_region, extract_text_with_vlm

    delay = float(os.environ.get("OCR_REQUEST_DELAY_SEC", "2.5"))
    if getattr(_ocr_crop_vlm, "_calls", 0) > 0 and delay > 0:  # type: ignore[attr-defined]
        time.sleep(delay)
    region = _prepare_ocr_region(crop.convert("RGB"))
    text = extract_text_with_vlm(_pil_to_base64(region))
    _ocr_crop_vlm._calls = getattr(_ocr_crop_vlm, "_calls", 0) + 1  # type: ignore[attr-defined]
    return text


def _load_pred_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("image_path")
            if key:
                out[key] = str(row.get("text") or "")
    return out


QWEN_OCR_PROMPT = "Transcribe the Persian text in this image. Output only the text."


def _qwen_load_key() -> tuple[str, str]:
    model_id = os.environ.get("QWEN_OCR_MODEL", _DEFAULT_QWEN_MODEL)
    lora = os.environ.get("QWEN_LORA_PATH", "").strip()
    return model_id, lora


def _ensure_qwen_local_loaded() -> tuple[object, object]:
    """Load (or reuse) Qwen + optional LoRA adapter. Returns (processor, model)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id, lora_path = _qwen_load_key()
    cache_key = f"{model_id}||{lora_path}"
    if getattr(_ensure_qwen_local_loaded, "_key", None) != cache_key:
        device = os.environ.get("QWEN_OCR_DEVICE", "mps")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float16, device_map=device
        )
        if lora_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_path)
            print(f"[baseline] loaded qwen_local {model_id} + LoRA {lora_path} on {device}")
        else:
            print(f"[baseline] loaded qwen_local {model_id} on {device}")
        model.eval()
        _ensure_qwen_local_loaded._key = cache_key  # type: ignore[attr-defined]
        _ensure_qwen_local_loaded._processor = processor  # type: ignore[attr-defined]
        _ensure_qwen_local_loaded._model = model  # type: ignore[attr-defined]
    return (
        _ensure_qwen_local_loaded._processor,  # type: ignore[attr-defined]
        _ensure_qwen_local_loaded._model,  # type: ignore[attr-defined]
    )


def _ocr_crop_qwen_local(crop: Image.Image) -> str:
    """Local Qwen3-VL Persian line OCR (bakeoff winner; optional LoRA)."""
    import torch

    processor, model = _ensure_qwen_local_loaded()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": crop.convert("RGB")},
            {"type": "text", "text": QWEN_OCR_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    new = out[0][inputs["input_ids"].shape[1] :]
    return processor.decode(new, skip_special_tokens=True).strip()


def _ocr_crops(
    crops: list[Image.Image],
    *,
    ocr_backend: str,
    image_paths: list[str] | None = None,
) -> list[str]:
    if ocr_backend == "surya":
        return _ocr_crops_surya(crops)
    if ocr_backend == "vlm":
        return [_ocr_crop_vlm(c) for c in crops]
    if ocr_backend == "qwen_local":
        # Bakeoff cache only for the base model without a LoRA adapter.
        _, lora_path = _qwen_load_key()
        if not lora_path:
            cache_path = Path(
                os.environ.get("QWEN_PRED_CACHE", str(_DEFAULT_QWEN_PRED_CACHE))
            )
            cache = _load_pred_cache(cache_path)
            if image_paths and all(p in cache for p in image_paths):
                return [cache[p] for p in image_paths]
        return [_ocr_crop_qwen_local(c) for c in crops]
    if ocr_backend == "empty":
        return [""] * len(crops)
    raise ValueError(f"unknown ocr backend: {ocr_backend}")


def _predict_line_crops(
    page_items: list[dict],
    *,
    ocr_backend: str,
    project_root: Path,
) -> list[tuple[dict, str, float | None]]:
    """OCR each labeled gold crop; return (page_item, pred_text, conf)."""
    results = []
    for item in page_items:
        labeled = [r for r in item["lines"] if (r.get("text") or "").strip()]
        if not labeled:
            results.append((item, "", None))
            continue
        crops = []
        paths = []
        for row in labeled:
            path = project_root / row["image_path"]
            crops.append(Image.open(path).convert("RGB"))
            paths.append(row["image_path"])
        texts = _ocr_crops(crops, ocr_backend=ocr_backend, image_paths=paths)
        results.append((item, "\n".join(texts), None))
    return results


def _load_bbox_index(project_root: Path) -> dict[str, list[int]]:
    """Map gold crop image_path → page-space bbox from lines_v3.jsonl."""
    path = project_root / "dataset" / "lines_v3.jsonl"
    index: dict[str, list[int]] = {}
    if not path.exists():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("image_path")
            bbox = row.get("crop_bbox") or row.get("bbox")
            if key and bbox and len(bbox) == 4:
                index[key] = [int(v) for v in bbox]
    return index


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _contains(outer: list[int], inner: list[int]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _match_score(gold_bbox: list[int], block_bbox: list[int]) -> float:
    """Score for assigning a gold line to a detected block.

    Gold ``crop_bbox`` values are often near full-page width (padding), while
    Surya line boxes hug RTL ink. Pure IoU then stays near zero even when the
    vertical band is correct — so we blend IoU with a vertical-band score.
    """
    iou = _iou(gold_bbox, block_bbox)
    gx1, gy1, gx2, gy2 = gold_bbox
    bx1, by1, bx2, by2 = block_bbox
    iy1, iy2 = max(gy1, by1), min(gy2, by2)
    oh = max(0, iy2 - iy1)
    ix1, ix2 = max(gx1, bx1), min(gx2, bx2)
    ow = max(0, ix2 - ix1)
    gh = max(1, gy2 - gy1)
    bh = max(1, by2 - by1)
    gw = max(1, gx2 - gx1)

    band = 0.0
    if oh > 0 and ow > 0:
        v_on_gold = oh / gh
        v_iou = oh / max(1, gh + bh - oh)
        gcy = 0.5 * (gy1 + gy2)
        bcy = 0.5 * (by1 + by2)
        y_close = max(0.0, 1.0 - abs(gcy - bcy) / max(gh, bh))
        h_ratio = min(gh, bh) / max(gh, bh)
        # Ignore width mismatch: gold may span the page; block may be short RTL.
        if v_on_gold >= 0.35 or y_close >= 0.6:
            band = 0.40 * v_on_gold + 0.30 * v_iou + 0.20 * y_close + 0.10 * h_ratio

    contained = 0.0
    if _contains(block_bbox, gold_bbox):
        tightness = (gh * gw) / max(1, (by2 - by1) * (bx2 - bx1))
        if tightness >= 0.25:
            contained = 0.55 + 0.45 * tightness
        else:
            contained = 0.05 * tightness

    return max(iou, band, contained)


def _assign_golds_to_blocks(
    gold_bboxes: list[list[int]],
    blocks: list[dict],
    *,
    min_score: float = 0.05,
) -> tuple[dict[int, int], list[dict]]:
    """Exclusive greedy matching, then allow multi-gold containment leftovers.

    Returns ``{gold_index: block_index}`` and the filtered text-block list.
    """
    # Include figure: Surya often labels handwriting lines as Figure/Picture.
    text_blocks = [
        b for b in blocks if b.get("type") in ("text", "list", "figure")
    ]
    if not text_blocks or not gold_bboxes:
        return {}, text_blocks

    pairs: list[tuple[float, int, int]] = []
    for gi, gb in enumerate(gold_bboxes):
        for bi, block in enumerate(text_blocks):
            bb = [int(v) for v in block["bbox"]]  # type: ignore[index]
            s = _match_score(gb, bb)
            if s >= min_score:
                pairs.append((s, gi, bi))
    pairs.sort(key=lambda t: t[0], reverse=True)

    assigned: dict[int, int] = {}
    used_blocks: set[int] = set()
    for score, gi, bi in pairs:
        if gi in assigned or bi in used_blocks:
            continue
        assigned[gi] = bi
        used_blocks.add(bi)

    # Leftover golds that sit inside a (possibly already-used) block — keep for
    # order-based line assignment within that block.
    for gi, gb in enumerate(gold_bboxes):
        if gi in assigned:
            continue
        best_bi, best_s = None, 0.0
        for bi, block in enumerate(text_blocks):
            bb = [int(v) for v in block["bbox"]]  # type: ignore[index]
            if not _contains(bb, gb):
                continue
            s = _match_score(gb, bb)
            if s > best_s:
                best_s, best_bi = s, bi
        if best_bi is not None and best_s >= min_score:
            assigned[gi] = best_bi

    return assigned, text_blocks


def _crop_for_gold(
    pil_page: Image.Image,
    gold_bbox: list[int],
    block_bbox: list[int],
) -> Image.Image:
    """Crop OCR region for one gold line.

    Use the full block when it is roughly line-height; otherwise take the
    block's horizontal span but the gold's vertical band (avoids feeding a
    multi-line region into a single-line reference comparison).
    """
    gx1, gy1, gx2, gy2 = gold_bbox
    bx1, by1, bx2, by2 = block_bbox
    gh = max(1, gy2 - gy1)
    bh = max(1, by2 - by1)
    if bh <= 2.5 * gh:
        return pil_page.crop((bx1, by1, bx2, by2))
    pad = max(2, int(0.15 * gh))
    y1 = max(by1, gy1 - pad)
    y2 = min(by2, gy2 + pad)
    if y2 <= y1:
        y1, y2 = by1, by2
    return pil_page.crop((bx1, y1, bx2, y2))


def _best_block_for_gold(
    gold_bbox: list[int],
    blocks: list[dict],
    *,
    min_iou: float = 0.05,
) -> dict | None:
    """Legacy single-gold helper (kept for diagnose overlays)."""
    text_blocks = [
        b for b in blocks if b.get("type") in ("text", "list", "figure")
    ]
    if not text_blocks:
        return None
    best, best_score = None, 0.0
    for block in text_blocks:
        bbox = [int(v) for v in block["bbox"]]  # type: ignore[index]
        score = _match_score(gold_bbox, bbox)
        if score > best_score:
            best_score, best = score, block
    if best is None or best_score < min_iou:
        return None
    return best


def _predict_pages(
    page_items: list[dict],
    *,
    ocr_backend: str,
    legacy_layout: bool,
    project_root: Path,
    match_records: list[dict] | None = None,
) -> list[tuple[dict, str, float | None, int]]:
    """Run layout + OCR on source pages, scored only on gold-labeled lines.

    Matching: exclusive greedy by ``_match_score``, then order-based assignment
    when multiple gold lines fall inside the same detected block.
    """
    from pipeline.layout import detect_blocks
    from pipeline.preprocess import enhance_page_rgb

    bbox_index = _load_bbox_index(project_root)
    out = []
    for item in page_items:
        page_path = project_root / item["page"]
        bgr = cv2.imread(str(page_path))
        if bgr is None:
            out.append((item, "", None, 0))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        page = enhance_page_rgb(rgb)
        pil_page = Image.fromarray(page)
        blocks = detect_blocks(page, legacy_full_blocks=legacy_layout)

        labeled = sorted(
            [r for r in item["lines"] if (r.get("text") or "").strip()],
            key=lambda r: r.get("line_index", 0),
        )
        if ocr_backend == "empty" or not labeled:
            out.append((item, "", None, len(blocks)))
            continue

        gold_bboxes: list[list[int] | None] = []
        for row in labeled:
            gold_bboxes.append(bbox_index.get(row["image_path"]))

        known_idx = [i for i, b in enumerate(gold_bboxes) if b is not None]
        known_bboxes = [gold_bboxes[i] for i in known_idx]  # type: ignore[misc]
        assigned, text_blocks = _assign_golds_to_blocks(
            known_bboxes,  # type: ignore[arg-type]
            blocks,
        )
        # Remap assigned indices (into known_bboxes) → labeled indices.
        labeled_assign: dict[int, int] = {
            known_idx[gi]: bi for gi, bi in assigned.items()
        }

        by_block: dict[int, list[int]] = defaultdict(list)
        for li, bi in labeled_assign.items():
            by_block[bi].append(li)

        preds: list[str | None] = [None] * len(labeled)
        confs: list[float] = [0.0] * len(labeled)
        crops_single: list[tuple[int, Image.Image]] = []

        for bi, line_idxs in by_block.items():
            block = text_blocks[bi]
            bb = [int(v) for v in block["bbox"]]  # type: ignore[index]
            conf = float(block.get("confidence") or 0.0)
            line_idxs_sorted = sorted(
                line_idxs,
                key=lambda i: (gold_bboxes[i] or [0, 0, 0, 0])[1],
            )
            if len(line_idxs_sorted) == 1:
                li = line_idxs_sorted[0]
                gb = gold_bboxes[li]
                assert gb is not None
                crop = _crop_for_gold(pil_page, gb, bb)
                if crop.size[0] < 2 or crop.size[1] < 2:
                    preds[li] = ""
                    confs[li] = 0.0
                else:
                    crops_single.append((li, crop))
                    confs[li] = conf
                if match_records is not None:
                    match_records.append(
                        {
                            "page": item["page"],
                            "image_path": labeled[li]["image_path"],
                            "ref": labeled[li].get("text") or "",
                            "gold_bbox": gb,
                            "block_bbox": bb,
                            "score": _match_score(gb, bb),
                            "n_share": 1,
                            "legacy_layout": legacy_layout,
                        }
                    )
            else:
                # Shared block: OCR once, split lines, assign by top-to-bottom order.
                crop = pil_page.crop((bb[0], bb[1], bb[2], bb[3]))
                if crop.size[0] < 2 or crop.size[1] < 2:
                    for li in line_idxs_sorted:
                        preds[li] = ""
                else:
                    texts = _ocr_crops([crop], ocr_backend=ocr_backend)
                    raw = texts[0] if texts else ""
                    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
                    # If OCR returned a single blob, fall back to per-gold band crops.
                    if len(lines) < len(line_idxs_sorted):
                        band_crops = []
                        for li in line_idxs_sorted:
                            gb = gold_bboxes[li]
                            assert gb is not None
                            band_crops.append(_crop_for_gold(pil_page, gb, bb))
                        lines = _ocr_crops(band_crops, ocr_backend=ocr_backend)
                    for j, li in enumerate(line_idxs_sorted):
                        preds[li] = lines[j] if j < len(lines) else ""
                        confs[li] = conf
                        if match_records is not None:
                            gb = gold_bboxes[li]
                            assert gb is not None
                            match_records.append(
                                {
                                    "page": item["page"],
                                    "image_path": labeled[li]["image_path"],
                                    "ref": labeled[li].get("text") or "",
                                    "gold_bbox": gb,
                                    "block_bbox": bb,
                                    "score": _match_score(gb, bb),
                                    "n_share": len(line_idxs_sorted),
                                    "legacy_layout": legacy_layout,
                                }
                            )

        if crops_single:
            order = [li for li, _ in crops_single]
            texts = _ocr_crops([c for _, c in crops_single], ocr_backend=ocr_backend)
            for li, text in zip(order, texts):
                preds[li] = text

        # Unmatched golds → empty (penalized) or gold-crop fallback only when no bbox.
        for i, row in enumerate(labeled):
            if preds[i] is not None:
                continue
            if gold_bboxes[i] is None:
                crop = Image.open(project_root / row["image_path"]).convert("RGB")
                texts = _ocr_crops([crop], ocr_backend=ocr_backend)
                preds[i] = texts[0] if texts else ""
                confs[i] = 0.0
                if match_records is not None:
                    match_records.append(
                        {
                            "page": item["page"],
                            "image_path": row["image_path"],
                            "ref": row.get("text") or "",
                            "gold_bbox": None,
                            "block_bbox": None,
                            "score": -1.0,
                            "n_share": 0,
                            "legacy_layout": legacy_layout,
                            "reason": "no_bbox_fallback_crop",
                        }
                    )
            else:
                preds[i] = ""
                confs[i] = 0.0
                if match_records is not None:
                    match_records.append(
                        {
                            "page": item["page"],
                            "image_path": row["image_path"],
                            "ref": row.get("text") or "",
                            "gold_bbox": gold_bboxes[i],
                            "block_bbox": None,
                            "score": 0.0,
                            "n_share": 0,
                            "legacy_layout": legacy_layout,
                            "reason": "no_match",
                        }
                    )

        mean_conf = sum(confs) / len(confs) if confs else None
        out.append((item, "\n".join(str(t or "") for t in preds), mean_conf, len(blocks)))
    return out


def _score_split(
    page_items: list[dict],
    predictions: list[tuple],
    *,
    split_name: str,
) -> tuple:
    pairs = []
    page_rows = []
    for item, *rest in predictions:
        pred = rest[0]
        conf = rest[1] if len(rest) > 1 else None
        ref = _page_reference(item)
        if not normalize_text(ref):
            continue
        pair = evaluate_pair(
            pred,
            ref,
            page=item["page"],
            region=item.get("source_document") or "",
            unit_mode="line",
            confidence=conf,
        )
        pairs.append(pair)
        page_rows.append(
            {
                "page": item["page"],
                "region": item.get("source_document") or "",
                "ref": ref,
                "pred": pred,
                "similarity": f"{pair.mean_sentence_similarity:.6f}",
                "align_tag": "page",
                "confidence": "" if conf is None else f"{conf:.4f}",
                "image_path": item["page"],
                "ground_truth": ref,
                "predicted": pred,
            }
        )

    note = (
        "LINE-LEVEL FALLBACK: gold labels are hand-verified line crops without "
        "reliable sentence punctuation (~11% of lines contain .؟!؛،). "
        "Units are newline-separated lines; sentence_similarity is still the "
        "primary partial-credit metric per aligned unit."
    )
    report = aggregate_results(pairs, unit_fallback_note=note)
    report.unit_mode_used = "line"
    return report, page_rows


def _write_report(
    report,
    page_rows: list[dict],
    out_dir: Path,
    *,
    tag: str,
    split_name: str,
    meta: dict,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"metrics_{split_name}_{tag}.json"
    csv_path = out_dir / f"errors_{split_name}_{tag}.csv"
    payload = report.to_dict()
    payload["meta"] = meta
    payload["split"] = split_name
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_error_csv(report.all_units, csv_path, extra_rows=None)

    all_rows = []
    # Attach per-line image paths when available from page manifests via meta.
    line_images = meta.get("line_image_paths") or {}
    for u in report.all_units:
        img = line_images.get((u.page, u.ref), u.page)
        all_rows.append(
            {
                "page": u.page,
                "region": u.region,
                "ref": u.ref,
                "pred": u.pred,
                "similarity": u.similarity,
                "align_tag": u.align_tag,
                "confidence": u.confidence,
                "image_path": img,
                "ground_truth": u.ref,
                "predicted": u.pred,
            }
        )
    for row in page_rows:
        r = dict(row)
        r["similarity"] = float(r["similarity"])
        r["align_tag"] = "page_aggregate"
        all_rows.append(r)
    all_rows.sort(key=lambda r: (float(r["similarity"]), r.get("page", "")))
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "page",
            "region",
            "ref",
            "pred",
            "similarity",
            "align_tag",
            "confidence",
            "image_path",
            "ground_truth",
            "predicted",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            out = dict(r)
            out["similarity"] = f"{float(out['similarity']):.6f}"
            if out.get("confidence") not in (None, ""):
                try:
                    out["confidence"] = f"{float(out['confidence']):.4f}"
                except (TypeError, ValueError):
                    pass
            else:
                out["confidence"] = ""
            w.writerow(out)
    return {"metrics": metrics_path, "errors": csv_path}


def _print_report(title: str, report) -> None:
    d = report.to_dict()
    print(f"\n=== {title} ===")
    print(f"  PRIMARY mean_sentence_similarity : {d['mean_sentence_similarity']:.4f}  (target ≥ 0.80)")
    boot_s = d.get("bootstrap_sentence_similarity") or {}
    if boot_s.get("n"):
        print(
            f"  bootstrap 95% CI (sentences)   : "
            f"[{boot_s['ci_low']:.4f}, {boot_s['ci_high']:.4f}]  n={boot_s['n']}  "
            f"iter={boot_s.get('n_boot', 1000)}"
        )
    boot_p = d.get("bootstrap_page_mean_similarity") or {}
    if boot_p.get("n"):
        print(
            f"  bootstrap 95% CI (pages)       : "
            f"[{boot_p['ci_low']:.4f}, {boot_p['ci_high']:.4f}]  n={boot_p['n']}"
        )
    print(f"  % units with similarity ≥ 0.80   : {100 * d['pct_similarity_ge_0_80']:.2f}%")
    print(f"  exact-match unit rate            : {100 * d['exact_match_sentence_rate']:.2f}%")
    print(f"  word_accuracy                    : {100 * d['word_accuracy']:.2f}%")
    print(f"  CER / WER                        : {d['cer']:.4f} / {d['wer']:.4f}")
    print(f"  pages / units                    : {d['n_pages']} / {d['n_units']}")
    print(f"  unit mode                        : {d['unit_mode_used']}")


def _line_image_index(page_items: list[dict]) -> dict[tuple[str, str], str]:
    """Map (page, ref_text) → crop image_path for error CSVs (best-effort)."""
    index: dict[tuple[str, str], str] = {}
    for item in page_items:
        page = item["page"]
        for row in item.get("lines", []):
            text = (row.get("text") or "").strip()
            if text and row.get("image_path"):
                index[(page, text)] = row["image_path"]
    return index


def run_debug_overlays(manifests: dict, out_dir: Path, n: int, project_root: Path) -> None:
    from pipeline.layout import save_debug_overlays

    pages = [project_root / p["page"] for p in manifests["val"]["pages"]]
    if len(pages) < n:
        pages += [project_root / p["page"] for p in manifests["train"]["pages"]]
    pages = pages[:n]
    line_dir = out_dir / "layout_debug_lines"
    leg_dir = out_dir / "layout_debug_legacy"
    print(f"[baseline] writing {len(pages)} line-level overlays → {line_dir}")
    save_debug_overlays(pages, line_dir, legacy_full_blocks=False, limit=n)
    print(f"[baseline] writing {len(pages)} legacy overlays → {leg_dir}")
    save_debug_overlays(pages, leg_dir, legacy_full_blocks=True, limit=n)


def _draw_match_overlay(
    page_bgr: np.ndarray,
    gold_bbox: list[int] | None,
    block_bbox: list[int] | None,
    *,
    label: str,
) -> np.ndarray:
    vis = page_bgr.copy()
    if block_bbox:
        x1, y1, x2, y2 = block_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
    if gold_bbox:
        x1, y1, x2, y2 = gold_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, label[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def diagnose_matching(
    page_items: list[dict],
    *,
    out_dir: Path,
    project_root: Path,
    ocr_backend: str = "surya",
    top_n: int = 10,
) -> dict:
    """Compare legacy vs line matching; write 10 worst overlays; re-score both."""
    match_line: list[dict] = []
    match_leg: list[dict] = []
    raw_line = _predict_pages(
        page_items,
        ocr_backend=ocr_backend,
        legacy_layout=False,
        project_root=project_root,
        match_records=match_line,
    )
    raw_leg = _predict_pages(
        page_items,
        ocr_backend=ocr_backend,
        legacy_layout=True,
        project_root=project_root,
        match_records=match_leg,
    )
    preds_line = [(a, b, c) for a, b, c, _ in raw_line]
    preds_leg = [(a, b, c) for a, b, c, _ in raw_leg]
    report_line, _ = _score_split(page_items, preds_line, split_name="val")
    report_leg, _ = _score_split(page_items, preds_leg, split_name="val")

    # Worst matches by score among line-layout assignments.
    scored = [r for r in match_line if r.get("score") is not None]
    scored.sort(key=lambda r: (float(r.get("score", 0.0)), r.get("ref", "")))
    worst = scored[:top_n]
    overlay_dir = out_dir / "match_overlays_worst10"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(worst):
        page_path = project_root / rec["page"]
        bgr = cv2.imread(str(page_path))
        if bgr is None:
            continue
        label = f"#{i} score={float(rec.get('score', 0)):.3f} share={rec.get('n_share', 0)}"
        vis = _draw_match_overlay(
            bgr,
            rec.get("gold_bbox"),
            rec.get("block_bbox"),
            label=label,
        )
        # Zoom to union of boxes when possible.
        boxes = [b for b in (rec.get("gold_bbox"), rec.get("block_bbox")) if b]
        if boxes:
            x1 = max(0, min(b[0] for b in boxes) - 20)
            y1 = max(0, min(b[1] for b in boxes) - 20)
            x2 = min(vis.shape[1], max(b[2] for b in boxes) + 20)
            y2 = min(vis.shape[0], max(b[3] for b in boxes) + 20)
            crop = vis[y1:y2, x1:x2]
        else:
            crop = vis
        out_path = overlay_dir / f"worst_{i:02d}_{Path(rec['image_path']).stem}.jpg"
        cv2.imwrite(str(out_path), crop)
        try:
            rec["overlay"] = str(out_path.resolve().relative_to(project_root.resolve()))
        except ValueError:
            rec["overlay"] = str(out_path)

    summary = {
        "line_layout": report_line.to_dict(),
        "legacy_layout": report_leg.to_dict(),
        "match_stats_line": {
            "n": len(match_line),
            "mean_score": (
                sum(float(r.get("score") or 0) for r in match_line) / len(match_line)
                if match_line
                else 0.0
            ),
            "n_share_gt1": sum(1 for r in match_line if int(r.get("n_share") or 0) > 1),
            "n_no_match": sum(1 for r in match_line if r.get("reason") == "no_match"),
        },
        "match_stats_legacy": {
            "n": len(match_leg),
            "mean_score": (
                sum(float(r.get("score") or 0) for r in match_leg) / len(match_leg)
                if match_leg
                else 0.0
            ),
            "n_share_gt1": sum(1 for r in match_leg if int(r.get("n_share") or 0) > 1),
            "n_no_match": sum(1 for r in match_leg if r.get("reason") == "no_match"),
        },
        "worst10": [
            {
                "score": r.get("score"),
                "n_share": r.get("n_share"),
                "ref": r.get("ref"),
                "page": r.get("page"),
                "image_path": r.get("image_path"),
                "overlay": r.get("overlay"),
                "reason": r.get("reason"),
            }
            for r in worst
        ],
    }
    path = out_dir / "matching_diagnosis.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[diagnose] line sim={report_line.mean_sentence_similarity:.4f}  "
          f"legacy sim={report_leg.mean_sentence_similarity:.4f}")
    print(f"[diagnose] overlays → {overlay_dir}")
    print(f"[diagnose] summary → {path}")
    return summary


def _print_comparison_table(results: dict[str, dict]) -> None:
    """Print surya vs vlm (and any other) comparison table."""
    print("\n========== BACKEND COMPARISON (line_crops) ==========")
    headers = ["backend", "split", "mean_sim", "%≥0.80", "CER", "WER", "word_acc", "n", "CI95"]
    print("  ".join(f"{h:>10s}" for h in headers))
    for key, d in results.items():
        boot = d.get("bootstrap_sentence_similarity") or {}
        ci = (
            f"[{boot.get('ci_low', 0):.3f},{boot.get('ci_high', 0):.3f}]"
            if boot.get("n")
            else "n/a"
        )
        row = [
            d.get("meta", {}).get("ocr_backend", key),
            d.get("split", ""),
            f"{d['mean_sentence_similarity']:.4f}",
            f"{100 * d['pct_similarity_ge_0_80']:.1f}%",
            f"{d['cer']:.4f}",
            f"{d['wer']:.4f}",
            f"{d['word_accuracy']:.4f}",
            str(boot.get("n", d.get("n_units", ""))),
            ci,
        ]
        print("  ".join(f"{c:>10s}" for c in row))


def run_error_analysis_from_report(
    report,
    page_items: list[dict],
    out_dir: Path,
    *,
    tag: str,
    split_name: str,
) -> Path:
    line_index = _line_image_index(page_items)
    rows = []
    for u in report.all_units:
        if u.align_tag == "insert":
            continue
        rows.append(
            {
                "page": u.page,
                "region": u.region,
                "ref": u.ref,
                "pred": u.pred,
                "similarity": u.similarity,
                "image_path": line_index.get((u.page, u.ref), u.page),
            }
        )
    analysis = analyze_units(rows, top_n=30)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"error_analysis_{split_name}_{tag}.json"
    csv_path = out_dir / f"error_analysis_worst30_{split_name}_{tag}.csv"
    summary_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_worst_csv(analysis["worst"], csv_path)
    print(f"\n=== Error analysis ({split_name}/{tag}) ===")
    print(f"  n_examples: {analysis['n_examples']}")
    for cat, count in analysis["category_counts"].items():
        if count:
            print(f"  {cat:24s} {count:4d}")
    print(f"  wrote {summary_path}")
    print(f"  wrote {csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=["val", "gold_test"])
    p.add_argument("--mode", choices=["page", "line_crops"], default="line_crops")
    p.add_argument("--ocr-backend", choices=list(OCR_BACKENDS), default="qwen_local")
    p.add_argument(
        "--compare-backends",
        nargs="+",
        choices=list(OCR_BACKENDS),
        default=None,
        help="Run multiple OCR backends on line_crops and print a comparison table.",
    )
    p.add_argument("--legacy-layout", action="store_true", help="Use pre-fix full-block layout")
    p.add_argument(
        "--compare-layout",
        action="store_true",
        help="Run legacy then line-level layout and print before/after metrics",
    )
    p.add_argument(
        "--diagnose-matching",
        action="store_true",
        help="Inspect IoU matching, write 10 worst overlays, compare legacy vs line.",
    )
    p.add_argument(
        "--error-analysis",
        action="store_true",
        help="After scoring, classify errors and write worst-30 CSV.",
    )
    p.add_argument(
        "--data-audit",
        action="store_true",
        help="Print gold data audit (and document leakage under current splits).",
    )
    p.add_argument("--debug-overlays", type=int, default=0, metavar="N")
    p.add_argument(
        "--overlays-only",
        action="store_true",
        help="With --debug-overlays, write overlays and exit (skip scoring)",
    )
    p.add_argument("--limit-pages", type=int, default=0)
    args = p.parse_args(argv)

    manifests = _ensure_splits(args.split_dir, args.gold, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_audit:
        audit = audit_gold(load_jsonl(args.gold))
        from eval.split import _document_leakage_check

        audit["document_leakage_under_split_dir"] = _document_leakage_check(manifests)
        audit_path = args.out_dir / "data_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print(f"[audit] wrote {audit_path}")
        return 0

    if args.debug_overlays:
        run_debug_overlays(manifests, args.out_dir, args.debug_overlays, PROJECT_ROOT)
        if args.overlays_only:
            return 0

    if args.diagnose_matching:
        items = [it for it in manifests["val"]["pages"] if it.get("n_labeled_lines", 0) > 0]
        if args.limit_pages:
            items = items[: args.limit_pages]
        diagnose_matching(
            items,
            out_dir=args.out_dir,
            project_root=PROJECT_ROOT,
            ocr_backend=args.ocr_backend,
        )
        return 0

    if args.compare_backends:
        if args.mode != "line_crops":
            print("[baseline] --compare-backends forces --mode line_crops")
            args.mode = "line_crops"
        comparison: dict[str, dict] = {}
        for backend in args.compare_backends:
            for split_name in args.splits:
                items = list(manifests[split_name]["pages"])
                if args.limit_pages:
                    items = items[: args.limit_pages]
                items = [it for it in items if it.get("n_labeled_lines", 0) > 0]
                tag = f"line_crops_{backend}"
                print(f"\n[baseline] split={split_name} pages={len(items)} backend={backend}")
                t0 = time.time()
                preds = _predict_line_crops(
                    items, ocr_backend=backend, project_root=PROJECT_ROOT
                )
                report, page_rows = _score_split(items, preds, split_name=split_name)
                meta = {
                    "mode": "line_crops",
                    "ocr_backend": backend,
                    "seconds": round(time.time() - t0, 2),
                    "line_image_paths": {
                        f"{k[0]}||{k[1]}": v
                        for k, v in _line_image_index(items).items()
                    },
                }
                # Store tuple keys separately for CSV writer
                paths = _write_report(
                    report,
                    page_rows,
                    args.out_dir,
                    tag=tag,
                    split_name=split_name,
                    meta={
                        "mode": "line_crops",
                        "ocr_backend": backend,
                        "seconds": round(time.time() - t0, 2),
                    },
                )
                # Fix image paths in error CSV for line crops
                _rewrite_error_csv_images(paths["errors"], items, report)
                _print_report(f"{split_name} / {tag}", report)
                if args.error_analysis and split_name == "gold_test":
                    run_error_analysis_from_report(
                        report, items, args.out_dir, tag=tag, split_name=split_name
                    )
                payload = report.to_dict()
                payload["meta"] = {"ocr_backend": backend}
                payload["split"] = split_name
                comparison[f"{backend}:{split_name}"] = payload
                print(f"  wrote {paths['metrics']}")
        _print_comparison_table(comparison)
        comp_path = args.out_dir / "backend_comparison.json"
        comp_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[baseline] comparison → {comp_path}")
        return 0

    if args.compare_layout:
        compare_splits = [s for s in args.splits if s != "gold_test"] or ["val"]
        if "gold_test" in args.splits and len(args.splits) == 1:
            compare_splits = ["gold_test"]
        summary_before_after = {}
        for split_name in compare_splits:
            items = list(manifests[split_name]["pages"])
            if args.limit_pages:
                items = items[: args.limit_pages]
            items = [it for it in items if it.get("n_labeled_lines", 0) > 0]
            print(f"\n[baseline] compare-layout split={split_name} pages={len(items)} mode={args.mode}")

            def _run(legacy: bool, tag: str):
                t0 = time.time()
                if args.mode == "line_crops":
                    preds = _predict_line_crops(
                        items, ocr_backend=args.ocr_backend, project_root=PROJECT_ROOT
                    )
                    n_blocks_note = "n/a (line_crops)"
                else:
                    raw = _predict_pages(
                        items,
                        ocr_backend=args.ocr_backend,
                        legacy_layout=legacy,
                        project_root=PROJECT_ROOT,
                    )
                    preds = [(a, b, c) for a, b, c, _n in raw]
                    n_blocks_note = float(np.mean([n for *_, n in raw])) if raw else 0
                report, page_rows = _score_split(items, preds, split_name=split_name)
                paths = _write_report(
                    report,
                    page_rows,
                    args.out_dir,
                    tag=tag,
                    split_name=split_name,
                    meta={
                        "mode": args.mode,
                        "ocr_backend": args.ocr_backend,
                        "legacy_layout": legacy,
                        "mean_blocks_per_page": n_blocks_note,
                        "seconds": round(time.time() - t0, 2),
                    },
                )
                _print_report(f"{split_name} / {tag}", report)
                print(f"  wrote {paths['metrics']}")
                print(f"  wrote {paths['errors']}")
                return report.to_dict()

            if args.mode == "line_crops":
                print(
                    "[baseline] WARNING: --mode line_crops ignores layout; "
                    "use --mode page for before/after layout comparison."
                )
                before = after = _run(False, "line_crops")
            else:
                before = _run(True, "before_legacy_layout")
                after = _run(False, "after_line_layout")
            summary_before_after[split_name] = {"before": before, "after": after}
            print("\n--- BEFORE → AFTER deltas ---")
            for key in (
                "mean_sentence_similarity",
                "word_accuracy",
                "cer",
                "wer",
                "exact_match_sentence_rate",
                "pct_similarity_ge_0_80",
            ):
                b, a = before[key], after[key]
                print(f"  {key:32s}  {b:.4f} → {a:.4f}  (Δ {a - b:+.4f})")

        summary_path = args.out_dir / "before_after_summary.json"
        summary_path.write_text(
            json.dumps(summary_before_after, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[baseline] summary → {summary_path}")
        _print_final_summary(summary_before_after)
        return 0

    # Single-pass evaluation
    last_report = None
    last_items = None
    last_tag = ""
    last_split = ""
    for split_name in args.splits:
        if split_name not in manifests:
            print(f"unknown split: {split_name}", file=sys.stderr)
            return 2
        items = list(manifests[split_name]["pages"])
        if args.limit_pages:
            items = items[: args.limit_pages]
        items = [it for it in items if it.get("n_labeled_lines", 0) > 0]
        tag = "legacy_layout" if args.legacy_layout else "line_layout"
        if args.mode == "line_crops":
            tag = f"line_crops_{args.ocr_backend}"
        print(f"\n[baseline] split={split_name} pages={len(items)} mode={args.mode} tag={tag}")
        t0 = time.time()
        if args.mode == "line_crops":
            preds = _predict_line_crops(
                items, ocr_backend=args.ocr_backend, project_root=PROJECT_ROOT
            )
        else:
            raw = _predict_pages(
                items,
                ocr_backend=args.ocr_backend,
                legacy_layout=args.legacy_layout,
                project_root=PROJECT_ROOT,
            )
            preds = [(a, b, c) for a, b, c, _n in raw]
        report, page_rows = _score_split(items, preds, split_name=split_name)
        paths = _write_report(
            report,
            page_rows,
            args.out_dir,
            tag=tag,
            split_name=split_name,
            meta={
                "mode": args.mode,
                "ocr_backend": args.ocr_backend,
                "legacy_layout": args.legacy_layout,
                "seconds": round(time.time() - t0, 2),
            },
        )
        if args.mode == "line_crops":
            _rewrite_error_csv_images(paths["errors"], items, report)
        _print_report(f"{split_name} / {tag}", report)
        print(f"  wrote {paths['metrics']}")
        print(f"  wrote {paths['errors']}")
        last_report, last_items, last_tag, last_split = report, items, tag, split_name
        if args.error_analysis:
            run_error_analysis_from_report(
                report, items, args.out_dir, tag=tag, split_name=split_name
            )

    return 0


def _rewrite_error_csv_images(csv_path: Path, page_items: list[dict], report) -> None:
    """Replace page paths with per-line crop paths in the error CSV."""
    index = _line_image_index(page_items)
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if row.get("align_tag") != "page_aggregate":
                key = (row.get("page") or "", row.get("ref") or "")
                if key in index:
                    row["image_path"] = index[key]
            rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print_final_summary(summary_before_after: dict) -> None:
    print("\n========== PHASE SUMMARY ==========")
    print("Dataset: dataset/gold.jsonl (hand-verified line crops; page-grouped splits).")
    print("Primary metric: mean sentence_similarity = 1 - lev(pred,ref)/max(len(ref),1), clipped.")
    print("Unit fallback: LINE-LEVEL (gold has sparse sentence punctuation).")
    print("gold_test: hold-out only — not used for tuning in this harness.")
    for split, ba in summary_before_after.items():
        b, a = ba["before"], ba["after"]
        print(
            f"{split}: sim {b['mean_sentence_similarity']:.3f}→{a['mean_sentence_similarity']:.3f} | "
            f"acc {b['word_accuracy']:.3f}→{a['word_accuracy']:.3f} | "
            f"CER {b['cer']:.3f}→{a['cer']:.3f} | WER {b['wer']:.3f}→{a['wer']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
