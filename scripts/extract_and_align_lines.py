#!/usr/bin/env python3
"""Build line-level training data where the label is bound to the crop BY CONSTRUCTION.

Replaces `scripts/extract_lines_legacy.py`, which paired Surya's detected boxes to
the page-level VLM's newline-separated output by list index after truncating to
`min(len(bboxes), len(label_lines))`. Surya detects ~31 boxes/page; the VLM emitted
~17 lines/page — so 96.5% of the resulting 17,186 rows carry another line's text.

The fix is structural, not a tuning change: **each crop is transcribed on its own.**
One crop in, one transcription out. There is no list to mis-zip.

    detect lines (Surya)  →  geometry filter  →  pad + crop  →  transcribe THIS crop  →  row

Crops are padded vertically by `PAD_FRACTION` per side before being written: the
detected boxes hug the ink, and v2 measured ink reaching the crop edge in 25.7%
(top) / 32.3% (bottom) of a 300-crop sample. The padding is applied to the crop
only, never to the geometry filter, so the set of accepted boxes is identical
with or without it. Rows carry both `bbox` (detected) and `crop_bbox` (written).

Duplicate page images are dropped before detection (see `dedupe_pages`): 45 groups
of byte-identical files exist in dataset/raw_images, and identical pixels under two
names would otherwise be free to land in two different splits.

Backends (`--backend`):
  none   free.  Detect, filter, and write crops + empty labels. Use this to prepare
                a hand-verification gold set, and to see the geometry report before
                spending anything.
  surya  free.  Local `surya_rec2` recognition (already in .cache/huggingface/hub).
                Try this before paying: if its CER against a hand-verified gold set
                is acceptable it labels the whole corpus for nothing.
  vlm    PAID.  One API call per crop. Refuses to run without --approve-api-spend;
                prints a cost estimate first either way.

Outputs:
  dataset/lines_v3.jsonl                — accepted rows
  dataset/lines_failed.jsonl            — every crop that failed, with the error.
                                          Failures are never silently dropped.
  dataset/geometry_filter_report.json   — per-rule drop counts
  dataset/lines_v3/                     — crop images

Usage:
  python scripts/extract_and_align_lines.py --backend none  --limit 20
  python scripts/extract_and_align_lines.py --backend surya
  python scripts/extract_and_align_lines.py --backend vlm --estimate-only
  python scripts/extract_and_align_lines.py --backend vlm --approve-api-spend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline.normalizer import (  # noqa: E402
    is_vlm_artifact,
    normalize_text,
    vlm_artifact_reason,
)

DATASET = PROJECT_ROOT / "dataset"
PAGES_JSONL = DATASET / "dataset.jsonl"      # source of page image paths
LINES_DIR = DATASET / "lines_v3"
LINES_OUT = DATASET / "lines_v3.jsonl"
FAILED_OUT = DATASET / "lines_failed.jsonl"
GEOMETRY_REPORT = DATASET / "geometry_filter_report.json"

# ── Crop padding ────────────────────────────────────────────────────────────
# Surya's boxes hug the ink, so descenders and dots fall outside them. Measured
# on 300 v2 crops with a page-level Otsu threshold: ink reaches the top edge in
# 25.7% and the bottom edge in 32.3%.
#
# PAD_FRACTION is per side, as a fraction of detected box height (0.075 → 15%
# total). It is applied to the CROP only, never to the geometry filter, so the
# set of boxes accepted stays identical to v2 and the two runs are comparable.
#
# Why not more: the clipped-ink distribution is bimodal. Contiguous ink outside
# the box is a small descender in the common case (median 0.16-0.23 of box
# height) but the tail runs past 1.0 — that tail is the *neighbouring line's*
# strokes, which padding should not chase. 11.4% of boxes already overlap a
# vertically-adjacent box, and at 0.075/side padding closes the gap to a
# neighbour in 16.6% of cases (vs 14.6% at 0.05, 22.2% at 0.15). 0.075 is the
# point where descenders are recovered without importing the next line.
PAD_FRACTION = 0.075

# ── Step 2: geometry filter thresholds ──────────────────────────────────────
MIN_HEIGHT_PX = 24     # 15.3% of legacy crops were shorter; 8px crops carry no strokes
MAX_ASPECT = 16.0      # 17.1% exceeded this; the tail reaches 250:1 (page rules, not text)
# Portrait crops (h > w) cannot be text lines: 3.4% of legacy crops were portrait.

# MIN_HEIGHT_PX is an ABSOLUTE pixel floor, so its effect depends on page
# resolution — and this corpus mixes two very different classes:
#   * 300-DPI PDF renders, median 2481x3508 → a text line is ~40-70px, ~15% dropped
#   * crawled web images, median ~500x700   → a text line is ~15-20px, ~80% dropped
# Both numbers are correct. The low-resolution crawled pages genuinely cannot yield
# trainable line crops: upscaling a 15px-tall line to TrOCR's 384px input invents
# stroke detail that was never captured. Excluding them is a data-quality decision,
# not a threshold to loosen — so the report breaks drops down by page size to keep
# that visible. Use --min-height to override if you disagree.
SMALL_PAGE_MAX_DIM = 1200  # pages whose long edge is below this are "low resolution"


@dataclass
class GeometryStats:
    min_height: int = MIN_HEIGHT_PX
    max_aspect: float = MAX_ASPECT
    detected: int = 0
    kept: int = 0
    drop_short: int = 0
    drop_portrait: int = 0
    drop_wide: int = 0
    drop_degenerate: int = 0
    per_page: dict = field(default_factory=dict)
    by_page_class: dict = field(default_factory=dict)

    def as_report(self) -> dict:
        d = self.detected or 1
        return {
            "thresholds": {
                "min_height_px": self.min_height,
                "max_aspect_ratio": self.max_aspect,
                "reject_portrait": True,
                "small_page_max_dim": SMALL_PAGE_MAX_DIM,
            },
            "detected_boxes": self.detected,
            "kept": self.kept,
            "kept_pct": round(100 * self.kept / d, 2),
            "dropped": {
                "height_below_min": self.drop_short,
                "portrait_orientation": self.drop_portrait,
                "aspect_above_max": self.drop_wide,
                "degenerate_zero_area": self.drop_degenerate,
            },
            "dropped_pct": {
                "height_below_min": round(100 * self.drop_short / d, 2),
                "portrait_orientation": round(100 * self.drop_portrait / d, 2),
                "aspect_above_max": round(100 * self.drop_wide / d, 2),
                "degenerate_zero_area": round(100 * self.drop_degenerate / d, 2),
            },
            "by_page_resolution": self.by_page_class,
            "pages": self.per_page,
        }

    def note_page_class(self, long_edge: int, detected: int, kept: int) -> None:
        cls = "low_res" if long_edge < SMALL_PAGE_MAX_DIM else "high_res"
        bucket = self.by_page_class.setdefault(
            cls, {"pages": 0, "detected": 0, "kept": 0}
        )
        bucket["pages"] += 1
        bucket["detected"] += detected
        bucket["kept"] += kept
        bucket["kept_pct"] = round(100 * bucket["kept"] / max(1, bucket["detected"]), 2)


def passes_geometry(x1: int, y1: int, x2: int, y2: int, stats: GeometryStats) -> bool:
    """Step 2 filter. Applied to the *bbox*, before any crop is written or any API
    call is made — this is the main cost lever (~36% of boxes never reach a label)."""
    w, h = x2 - x1, y2 - y1
    stats.detected += 1
    if w <= 0 or h <= 0:
        stats.drop_degenerate += 1
        return False
    if h < stats.min_height:
        stats.drop_short += 1
        return False
    if h > w:
        stats.drop_portrait += 1
        return False
    if w / h > stats.max_aspect:
        stats.drop_wide += 1
        return False
    stats.kept += 1
    return True


# ── Document identity ───────────────────────────────────────────────────────
# Split grouping (Step 4) needs a stable document id. PDF page renders are named
# "<document>_pNNN.png" by scripts/ingest_pdfs.py; crawled images are bare UUIDs.
_PAGE_SUFFIX_RE = re.compile(r"_p\d{1,4}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def derive_source_document(image_path: str | Path) -> str:
    """Stable parent-document id for a page image.

    Every crawled image becomes its own single-page 'document' rather than one
    shared WEB_CRAWLED bucket: they come from unrelated sources with unrelated
    writers, so bucketing them together would let the same writer's pages cross
    splits — the exact failure this function exists to prevent.
    """
    stem = Path(image_path).stem
    if _UUID_RE.match(stem):
        return f"web/{stem}"
    doc = _PAGE_SUFFIX_RE.sub("", stem)
    return doc or stem


def detect_line_bboxes(
    rgb: np.ndarray, *, min_height: int = MIN_HEIGHT_PX
) -> list[tuple[int, int, int, int]]:
    """Surya text-line detection, returned as clipped integer xyxy in page space."""
    from pipeline.layout import det_model, det_processor  # lazy: heavy import
    from surya.detection import batch_text_detection

    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb)
    preds = batch_text_detection([pil], det_model, det_processor)
    if not preds:
        return []
    out = []
    for box in preds[0].bboxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box.bbox)
        x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
        x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
        out.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    # Top-to-bottom by band, then right-to-left within a band (Persian reading order).
    # Banding on min_height rather than exact y1 so two boxes on the same visual line
    # actually group together — the legacy sort keyed on exact y1, so its RTL
    # secondary key almost never fired.
    band = max(1, min_height)
    out.sort(key=lambda b: (b[1] // band, -b[2]))
    return out


# ── Label backends: one crop in, one transcription out ──────────────────────

_VLM_SYSTEM_PROMPT = (
    "You are an OCR/HTR engine for Persian (Farsi). The image is a SINGLE cropped "
    "line from a scanned handwritten or printed exam/notebook page. Transcribe "
    "exactly the text visible in this one line. Output only the transcription: no "
    "preamble, no translation, no markdown, no table syntax, no bullet points, and "
    "no commentary. Preserve digits, punctuation, and parentheses as written. If the "
    "line is entirely illegible, output exactly: [ILLEGIBLE]"
)
_VLM_USER_PROMPT = "Transcribe this single line of Persian text."
ILLEGIBLE_MARKER = "[ILLEGIBLE]"


def label_with_surya(crops: list[Image.Image]) -> list[str]:
    """Batch-recognize crops with the local surya_rec2 model. Free, no network."""
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from surya.recognition import batch_recognition

    global _REC_MODEL, _REC_PROCESSOR
    if _REC_MODEL is None:
        _REC_MODEL = load_rec_model()
        _REC_PROCESSOR = load_rec_processor()
    langs = [["fa"]] * len(crops)
    preds = batch_recognition(crops, langs, _REC_MODEL, _REC_PROCESSOR)
    texts = preds[0] if isinstance(preds, tuple) else preds
    return [str(t or "") for t in texts]


_REC_MODEL = None
_REC_PROCESSOR = None
_VLM_CLIENT = None


def label_with_vlm(crop: Image.Image, *, model: str, delay: float) -> str:
    """One API call for one crop. The label cannot belong to another line."""
    import base64
    import io

    from pipeline.openai_compat import create_openai_client

    global _VLM_CLIENT
    if _VLM_CLIENT is None:
        _VLM_CLIENT = create_openai_client()

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = _VLM_CLIENT.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=256,
        messages=[
            # English system prompt: pipeline/openai_compat.py:17 documents that the
            # AgentRouter edge filter content-blocks Persian system prompts, and .env
            # currently points VLM_API_BASE there.
            {"role": "system", "content": _VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": _VLM_USER_PROMPT},
                ],
            },
        ],
    )
    if delay:
        time.sleep(delay)
    return (resp.choices[0].message.content or "").strip()


# ── Cost estimation (guardrail: no paid call without an explicit estimate) ───
# Anthropic image tokens ≈ (width × height) / 750.
PRICING_USD_PER_MTOK = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
PROMPT_TOKENS = 130      # system + user prompt
OUTPUT_TOKENS = 35       # a single transcribed line


def estimate_cost(boxes: list[tuple[int, int, int, int]]) -> dict:
    image_tokens = sum(((x2 - x1) * (y2 - y1)) / 750 for x1, y1, x2, y2 in boxes)
    n = len(boxes)
    in_tok = image_tokens + n * PROMPT_TOKENS
    out_tok = n * OUTPUT_TOKENS
    rows = {}
    for name, (pin, pout) in PRICING_USD_PER_MTOK.items():
        usd = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        rows[name] = {"standard_usd": round(usd, 2), "batch_api_usd": round(usd / 2, 2)}
    return {
        "calls": n,
        "input_tokens": int(in_tok),
        "image_tokens": int(image_tokens),
        "output_tokens": int(out_tok),
        "by_model": rows,
        "caveats": [
            "Prompt caching does not apply: the ~130-token system prompt is far "
            "below the 4096-token minimum cacheable prefix on Opus-tier models.",
            "Assumes the first-party Anthropic API. A third-party router "
            "(JustWoker, AgentRouter, …) may bill at different rates.",
        ],
    }


def print_cost_estimate(est: dict) -> None:
    print("\n" + "=" * 68)
    print("  API COST ESTIMATE — no paid call has been made yet")
    print("=" * 68)
    print(f"  crops to label : {est['calls']:,}")
    print(f"  input tokens   : {est['input_tokens']:,}  "
          f"(of which {est['image_tokens']:,} are image tokens)")
    print(f"  output tokens  : {est['output_tokens']:,}")
    print(f"\n  {'model':22s} {'standard':>10s} {'batch -50%':>12s}")
    for name, v in est["by_model"].items():
        print(f"  {name:22s} {'$' + str(v['standard_usd']):>10s} "
              f"{'$' + str(v['batch_api_usd']):>12s}")
    for c in est["caveats"]:
        print(f"\n  ! {c}")
    print("=" * 68)


# ── Page enumeration and crop extraction ────────────────────────────────────


def load_page_paths(pages_jsonl: Path, images_dir: Path | None) -> list[Path]:
    """Enumerate source pages. Deliberately does NOT read `text` from the manifest —
    page-level labels are not used at all any more."""
    if images_dir:
        return sorted(
            p for p in images_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        )
    seen: dict[str, Path] = {}
    with pages_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel = json.loads(line).get("image_path")
            if rel and rel not in seen:
                seen[rel] = PROJECT_ROOT / rel
    return [p for p in seen.values() if p.exists()]


def dedupe_pages(paths: list[Path]) -> tuple[list[Path], dict]:
    """Drop byte-identical duplicate page images, keeping the first path by name.

    45 groups of identical files exist in dataset/raw_images (63 redundant copies,
    42 of the groups crawled web images saved under different UUIDs). This has to
    be caught here rather than in the split: `derive_source_document` gives every
    UUID page its own document id, so two identical files become two distinct
    "documents" that the bin-packer can place in different splits — the same
    pixels in train and test, with a leakage report that correctly shows zero
    shared documents. Content-addressed crop names do not catch it either, since
    they hash `source_image`, which differs.
    """
    by_hash: dict[str, Path] = {}
    dropped: list[tuple[str, str]] = []
    for path in sorted(paths):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            by_hash[f"unreadable:{path}"] = path
            continue
        if digest in by_hash:
            dropped.append((_rel(path), _rel(by_hash[digest])))
        else:
            by_hash[digest] = path
    return sorted(by_hash.values()), {
        "pages_in": len(paths),
        "pages_unique": len(by_hash),
        "duplicate_pages_dropped": len(dropped),
        "examples": [{"dropped": d, "kept": k} for d, k in dropped[:10]],
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _rel(path: Path) -> str:
    """Repo-root-relative POSIX path — the form every JSONL `image_path` uses."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def scan_pages(
    paths: list[Path], stats: GeometryStats, *, pad: float = PAD_FRACTION
) -> list[dict]:
    """Detect + geometry-filter every page. No labelling, no API calls, no writes.

    Padding is resolved here, while the page dimensions are in hand, so the cost
    estimate and the crop writer both see the same `crop_bbox`.
    """
    plan: list[dict] = []
    for i, page in enumerate(paths, 1):
        rel = _rel(page)
        try:
            rgb = load_rgb(page)
        except Exception as exc:
            stats.per_page[rel] = {"error": str(exc)}
            continue
        boxes = detect_line_bboxes(rgb, min_height=stats.min_height)
        before = stats.detected
        kept = [b for b in boxes if passes_geometry(*b, stats)]
        detected = stats.detected - before
        page_h, page_w = rgb.shape[:2]
        stats.per_page[rel] = {
            "width": int(page_w),
            "height": int(page_h),
            "detected": detected,
            "kept": len(kept),
        }
        stats.note_page_class(max(rgb.shape[:2]), detected, len(kept))
        for idx, box in enumerate(kept):
            plan.append({
                "page": page,
                "source_image": rel,
                "source_document": derive_source_document(page),
                "line_index": idx,
                "bbox": list(box),
                "crop_bbox": list(
                    pad_bbox(box, page_w=page_w, page_h=page_h, fraction=pad)
                ),
            })
        if i % 25 == 0 or i == len(paths):
            print(f"  scanned {i}/{len(paths)} pages · "
                  f"{stats.detected} boxes → {stats.kept} kept", flush=True)
    return plan


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pad_bbox(
    bbox: list[int] | tuple[int, int, int, int],
    *,
    page_w: int,
    page_h: int,
    fraction: float = PAD_FRACTION,
) -> tuple[int, int, int, int]:
    """Grow a box vertically by `fraction` of its height on each side, clamped
    to the page. Horizontal extent is untouched: the clipping measured in v2 is
    vertical (descenders and dots), and Persian lines sit close enough
    side-by-side that widening pulls in adjacent text.
    """
    x1, y1, x2, y2 = bbox
    dy = int(round(fraction * (y2 - y1)))
    return x1, max(0, y1 - dy), x2, min(page_h, y2 + dy)


def crop_name(source_image: str, bbox: list[int]) -> str:
    """Content-addressed crop filename: same page + same box → same file.

    Was `uuid4().hex[:12]`, which made every re-run write a fresh copy of every
    crop and leave the previous run's files behind as orphans in dataset/lines_v2/
    (the probe runs above accumulated 64 of them). Hashing the identity of the
    crop instead makes the extraction idempotent and lets a re-run overwrite in
    place, so the directory always matches the JSONL.
    """
    key = f"{source_image}|{','.join(str(v) for v in bbox)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12] + ".jpg"


def run_labelling(plan: list[dict], args) -> tuple[list[dict], list[dict], Counter]:
    """Crop, label, validate. Returns (accepted, failed, reason_counts).

    Nothing is dropped invisibly: every crop lands in `accepted` or `failed`.
    """
    lines_dir = args.lines_dir
    lines_rel = _rel(lines_dir)
    lines_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[dict] = []
    failed: list[dict] = []
    reasons: Counter = Counter()

    by_page: dict[str, list[dict]] = {}
    for item in plan:
        by_page.setdefault(item["source_image"], []).append(item)

    for pi, (rel, items) in enumerate(by_page.items(), 1):
        try:
            rgb = load_rgb(PROJECT_ROOT / rel)
        except Exception as exc:
            for item in items:
                failed.append({**_row_meta(item), "error": f"page_load: {exc}"})
                reasons["page_load_error"] += 1
            continue

        page_h, page_w = rgb.shape[:2]
        crops, metas = [], []
        for item in items:
            # `bbox` stays the detected box — it is what the geometry filter
            # judged and what makes v2/v3 comparable. `crop_bbox` is what was
            # actually written to disk, recorded so the padding is auditable
            # and a later re-crop can reproduce it exactly.
            crop_box = tuple(item["crop_bbox"])
            x1, y1, x2, y2 = crop_box
            crop = Image.fromarray(rgb[y1:y2, x1:x2])
            name = crop_name(item["source_image"], list(crop_box))
            crop.save(lines_dir / name, quality=95)
            crops.append(crop)
            metas.append({**_row_meta(item),
                          "image_path": f"{lines_rel}/{name}"})

        texts = _transcribe(crops, metas, failed, reasons, args)
        for meta, text in zip(metas, texts):
            if text is None:
                continue
            if args.backend == "none":
                # Crop-only mode: emit an empty label for a human to fill in.
                accepted.append({**meta, "text": "", "aligned": True,
                                 "label_backend": "none"})
                reasons["crop_only"] += 1
                continue
            reason = _validate(text)
            if reason:
                failed.append({**meta, "text_raw": text, "error": f"rejected: {reason}"})
                reasons[reason] += 1
                continue
            accepted.append({**meta, "text": normalize_text(text), "aligned": True,
                             "label_backend": args.backend})
            reasons["accepted"] += 1

        if pi % 10 == 0 or pi == len(by_page):
            print(f"  labelled {pi}/{len(by_page)} pages · "
                  f"{len(accepted)} accepted, {len(failed)} failed", flush=True)
    return accepted, failed, reasons


def _row_meta(item: dict) -> dict:
    return {
        "source_image": item["source_image"],
        "source_document": item["source_document"],
        "line_index": item["line_index"],
        "bbox": item["bbox"],
        "crop_bbox": item["crop_bbox"],
    }


def _validate(text: str) -> str | None:
    """Return a rejection reason, or None to accept."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty_transcription"
    if stripped == ILLEGIBLE_MARKER:
        return "model_reported_illegible"
    if is_vlm_artifact(stripped):
        return f"artifact_{vlm_artifact_reason(stripped)}"
    if not normalize_text(stripped):
        return "empty_after_normalization"
    return None


def _transcribe(crops, metas, failed, reasons, args) -> list[str | None]:
    """Dispatch to the chosen backend. Per-crop errors become `failed` rows."""
    if args.backend == "none":
        return [""] * len(crops)

    if args.backend == "surya":
        try:
            return label_with_surya(crops)
        except Exception as exc:
            for meta in metas:
                failed.append({**meta, "error": f"surya_batch: {exc}"})
                reasons["surya_batch_error"] += 1
            return [None] * len(crops)

    out: list[str | None] = []
    for meta, crop in zip(metas, crops):
        try:
            out.append(label_with_vlm(crop, model=args.model, delay=args.delay))
        except Exception as exc:
            failed.append({**meta, "error": f"vlm: {type(exc).__name__}: {exc}"})
            reasons["vlm_error"] += 1
            out.append(None)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Build line data with labels bound to crops by construction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", choices=["none", "surya", "vlm"], default="none")
    p.add_argument("--pages-jsonl", type=Path, default=PAGES_JSONL)
    p.add_argument("--images-dir", type=Path, default=None,
                   help="Scan this directory instead of --pages-jsonl")
    p.add_argument("--limit", type=int, default=0, help="Only the first N pages")
    p.add_argument("--keep-duplicate-pages", action="store_true",
                   help="Skip the byte-identical page dedup pass (not recommended: "
                        "identical pages under different names can otherwise be "
                        "split across train and test)")
    p.add_argument("--min-height", type=int, default=MIN_HEIGHT_PX,
                   help=f"Drop crops shorter than this many pixels "
                        f"(default {MIN_HEIGHT_PX}). Resolution-dependent — see the "
                        f"by_page_resolution section of the geometry report.")
    p.add_argument("--max-aspect", type=float, default=MAX_ASPECT,
                   help=f"Drop crops wider than this many times their height "
                        f"(default {MAX_ASPECT})")
    p.add_argument("--model", default=os.environ.get("VLM_MODEL", "claude-opus-4-8"))
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds between VLM calls (rate-limit courtesy)")
    p.add_argument("--estimate-only", action="store_true",
                   help="Scan + geometry-filter + print the cost estimate, then stop")
    p.add_argument("--approve-api-spend", action="store_true",
                   help="Required for --backend vlm. Without it the script stops "
                        "after printing the estimate.")
    p.add_argument("--out", type=Path, default=LINES_OUT)
    p.add_argument("--lines-dir", type=Path, default=LINES_DIR,
                   help=f"Directory for crop images (default {LINES_DIR.name})")
    p.add_argument("--pad", type=float, default=PAD_FRACTION,
                   help=f"Vertical crop padding per side, as a fraction of "
                        f"detected box height (default {PAD_FRACTION} → "
                        f"{2 * PAD_FRACTION:.0%} total). Applied to the crop only, "
                        f"never to the geometry filter. Use 0 to reproduce v2.")
    p.add_argument("--failed-out", type=Path, default=FAILED_OUT)
    p.add_argument("--geometry-report", type=Path, default=GEOMETRY_REPORT,
                   help="Override so probe runs do not clobber the corpus report")
    args = p.parse_args()

    if args.images_dir is None and not args.pages_jsonl.exists():
        print(f"Not found: {args.pages_jsonl}")
        return 1

    pages = load_page_paths(args.pages_jsonl, args.images_dir)
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        print("No source pages found.")
        return 1
    dedup = {"pages_in": len(pages), "pages_unique": len(pages),
             "duplicate_pages_dropped": 0, "examples": []}
    if not args.keep_duplicate_pages:
        pages, dedup = dedupe_pages(pages)
        if dedup["duplicate_pages_dropped"]:
            print(f"Dropped {dedup['duplicate_pages_dropped']} byte-identical "
                  f"duplicate page image(s) — identical pixels must not be able to "
                  f"land in two different splits.")
    print(f"Source pages: {len(pages)}   backend={args.backend}   "
          f"crop pad={args.pad:.3g}/side ({2 * args.pad:.0%} total)")

    print("\nStep 1+2 — detect lines and apply the geometry filter…")
    stats = GeometryStats(min_height=args.min_height, max_aspect=args.max_aspect)
    plan = scan_pages(pages, stats, pad=args.pad)
    report = stats.as_report()
    report["page_deduplication"] = dedup
    report["crop_padding"] = {
        "fraction_per_side": args.pad,
        "fraction_total": round(2 * args.pad, 4),
        "axis": "vertical",
        "applied_to": "crop only, not the geometry filter",
    }
    args.geometry_report.parent.mkdir(parents=True, exist_ok=True)
    args.geometry_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n  detected boxes            : {report['detected_boxes']:,}")
    for rule, n in report["dropped"].items():
        print(f"  dropped {rule:24s}: {n:,} "
              f"({report['dropped_pct'][rule]}%)")
    print(f"  kept                      : {report['kept']:,} "
          f"({report['kept_pct']}%)")
    if len(report["by_page_resolution"]) > 1:
        print("\n  keep rate by page resolution (the min-height threshold is"
              " absolute, so this varies):")
        for cls, v in sorted(report["by_page_resolution"].items()):
            print(f"    {cls:9s} {v['pages']:4d} pages  "
                  f"{v['detected']:6d} boxes → {v['kept']:6d} kept "
                  f"({v['kept_pct']}%)")
    print(f"\n  report → {args.geometry_report}")

    # Estimated on `crop_bbox`, not `bbox` — the padded crop is what gets uploaded,
    # so it is what the image tokens are billed on.
    est = estimate_cost([tuple(i["crop_bbox"]) for i in plan])
    if args.backend == "vlm":
        print_cost_estimate(est)
        if not args.approve_api_spend or args.estimate_only:
            print("\n  Stopping before any paid call.")
            print("  Re-run with --approve-api-spend to proceed, or try the free "
                  "local backend first:\n"
                  "      python scripts/extract_and_align_lines.py --backend surya")
            return 0
    elif args.estimate_only:
        print_cost_estimate(est)
        return 0

    print(f"\nStep 1 — transcribing {len(plan):,} crops individually…")
    accepted, failed, reasons = run_labelling(plan, args)

    write_jsonl(args.out, accepted)
    write_jsonl(args.failed_out, failed)

    print(f"\n  accepted : {len(accepted):,} → {args.out}")
    print(f"  failed   : {len(failed):,} → {args.failed_out}")
    print("\n  outcome breakdown:")
    for reason, n in reasons.most_common():
        print(f"    {reason:34s} {n:,}")
    print(f"\n  crops → {args.lines_dir}")
    try:
        out_rel = args.out.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        out_rel = str(args.out)
    print(f"\nNext: python scripts/split_dataset.py --input {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
