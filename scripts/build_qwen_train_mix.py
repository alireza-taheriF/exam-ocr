#!/usr/bin/env python3
"""Build a leakage-safe training mix for Qwen LoRA fine-tuning.

Sources (document-split train docs only for in-domain crops):
  1. Hand-labeled gold lines from train documents
  2. Pseudo-labels from lines_v3 on the same train documents (qwen_local)
  3. External printed Persian OCR (persian-ocr-230k)
  4. Oversample short / number-heavy gold rows

Outputs:
  dataset/qwen_ft/train_mix.jsonl
  dataset/qwen_ft/val_gold.jsonl   (gold val only — never pseudo)

Usage:
  python scripts/build_qwen_train_mix.py
  python scripts/build_qwen_train_mix.py --pseudo-limit 800 --external-limit 5000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import is_vlm_artifact  # noqa: E402

DEFAULT_SPLIT_DIR = PROJECT_ROOT / "eval" / "splits_by_document"
DEFAULT_LINES_V3 = PROJECT_ROOT / "dataset" / "lines_v3.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "dataset" / "qwen_ft"
DEFAULT_PSEUDO_CACHE = DEFAULT_OUT_DIR / "pseudo_preds.jsonl"

_DIGIT_RE = re.compile(r"[0-9۰-۹٠-٩]")
_PUNCT_RE = re.compile(r"[.,?!؛،:()\[\]{}|/\\+\-=%#@]")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_docs(split_dir: Path, split: str) -> set[str]:
    manifest = json.loads((split_dir / f"{split}.json").read_text(encoding="utf-8"))
    return {p.get("source_document") or "" for p in manifest["pages"]}


def _gold_rows(split_dir: Path, split: str) -> list[dict]:
    manifest = json.loads((split_dir / f"{split}.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for page in manifest["pages"]:
        for row in page.get("lines", []):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "image_path": row["image_path"],
                    "text": text,
                    "source": "gold",
                    "source_document": page.get("source_document"),
                    "source_image": page.get("page") or row.get("source_image"),
                    "split": split,
                }
            )
    return out


def _is_number_heavy(text: str) -> bool:
    if not text:
        return False
    digits = len(_DIGIT_RE.findall(text))
    punct = len(_PUNCT_RE.findall(text))
    return digits + punct >= max(2, len(text) // 4)


def _is_short(text: str) -> bool:
    return 0 < len(text.strip()) <= 12


def _accept_pseudo(text: str, *, min_len: int = 2) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < min_len:
        return False
    if is_vlm_artifact(stripped):
        return False
    # Reject digit-spam / collapse failures from the teacher model.
    if len(stripped) >= 20 and len(set(stripped)) <= 2 and _DIGIT_RE.search(stripped):
        return False
    if stripped.count("۰") >= 12 or stripped.count("0") >= 12:
        return False
    return True


def _pseudo_label_train_docs(
    *,
    lines_v3: Path,
    train_docs: set[str],
    gold_paths: set[str],
    cache_path: Path,
    limit: int,
    project_root: Path,
) -> list[dict]:
    """Run qwen_local on unlabeled train-doc crops; resume via cache_path."""
    from PIL import Image

    from eval.run_baseline import _ocr_crop_qwen_local

    candidates: list[dict] = []
    with lines_v3.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source_document") not in train_docs:
                continue
            ip = row.get("image_path") or ""
            if not ip or ip in gold_paths:
                continue
            if not (project_root / ip).exists():
                continue
            candidates.append(row)

    cache = {r["image_path"]: r for r in load_jsonl(cache_path)}
    todo = [r for r in candidates if r["image_path"] not in cache]
    if limit > 0:
        # Cap how many NEW inferences we run; accepted filter applied after.
        todo = todo[: max(0, limit - len(cache))]

    print(f"[mix] pseudo candidates={len(candidates)} cached={len(cache)} todo={len(todo)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            img = Image.open(project_root / row["image_path"]).convert("RGB")
            try:
                text = _ocr_crop_qwen_local(img)
            except Exception as exc:  # noqa: BLE001
                text = ""
                err = f"{type(exc).__name__}: {exc}"
            else:
                err = ""
            entry = {
                "image_path": row["image_path"],
                "text": text,
                "source": "pseudo_qwen",
                "source_document": row.get("source_document"),
                "source_image": row.get("source_image"),
                "error": err,
            }
            cache[row["image_path"]] = entry
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sink.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[mix] pseudo labeled {i}/{len(todo)}")

    accepted: list[dict] = []
    for row in candidates:
        entry = cache.get(row["image_path"])
        if not entry:
            continue
        text = (entry.get("text") or "").strip()
        if not _accept_pseudo(text):
            continue
        accepted.append(
            {
                "image_path": entry["image_path"],
                "text": text,
                "source": "pseudo_qwen",
                "source_document": entry.get("source_document"),
                "source_image": entry.get("source_image"),
            }
        )
        if limit > 0 and len(accepted) >= limit:
            break
    print(f"[mix] pseudo accepted={len(accepted)}")
    return accepted


def _generate_synthetic_lines(
    *,
    limit: int,
    out_dir: Path,
    seed: int,
) -> list[dict]:
    """Render Persian printed lines (numbers/punct-heavy) with ReportLab."""
    if limit <= 0:
        return []
    from io import BytesIO

    import pypdfium2 as pdfium
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/GeezaPro.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/GeezaPro.ttc",
    ]
    font_path = next((p for p in font_candidates if Path(p).exists()), None)
    if font_path is None:
        print("[mix] synthetic skipped: no Persian-capable system font found")
        return []

    try:
        pdfmetrics.registerFont(TTFont("FaSynth", font_path))
        font_name = "FaSynth"
    except Exception as exc:  # noqa: BLE001
        print(f"[mix] synthetic font register failed: {exc}")
        return []

    rng = random.Random(seed)
    templates = [
        "بدهکار {a} ماه",
        "بستانکار {a}",
        "جمع کل: {a}",
        "{a} {b}",
        "سال {a}",
        "ص {a}",
        "مبلغ {a} ریال",
        "y=f(a) ; ({b}) = (y=0)",
        "فصل {a}",
        "دارایی جاری ({a} سال یا کمتر)",
        "آب، برق، تلفن",
        "۹۷۲۸۰۰۰۰ ۲۵۵۰۰۰۰",
        "Subject:",
        "Year: {a}",
    ]
    img_dir = out_dir / "images" / "synthetic"
    img_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for i in range(limit):
        tmpl = templates[i % len(templates)]
        a = "".join(rng.choice("۰۱۲۳۴۵۶۷۸۹") for _ in range(rng.randint(1, 8)))
        b = "".join(rng.choice("۰۱۲۳۴۵۶۷۸۹") for _ in range(rng.randint(1, 6)))
        text = tmpl.format(a=a, b=b)
        # Render one-line PDF page then rasterize.
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=(900, 80))
        c.setFont(font_name, 22)
        c.drawRightString(880, 30, text)
        c.save()
        pdf = pdfium.PdfDocument(buf.getvalue())
        page = pdf[0]
        pil = page.render(scale=2).to_pil().convert("RGB")
        rel = f"dataset/qwen_ft/images/synthetic/syn_{i:05d}.jpg"
        abs_path = PROJECT_ROOT / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(abs_path, quality=92)
        records.append(
            {
                "image_path": rel,
                "text": text,
                "source": "synthetic_print",
            }
        )
    print(f"[mix] synthetic generated={len(records)}")
    return records


def _load_or_download_external(*, limit: int, project_root: Path) -> list[dict]:
    import importlib.util

    out_dir = project_root / "dataset" / "external" / "persian-ocr-230k"
    meta = out_dir / "meta.json"
    records_path = out_dir / "records.jsonl"
    if records_path.exists():
        rows = load_jsonl(records_path)
        if limit > 0:
            rows = rows[:limit]
        print(f"[mix] external from cache={len(rows)}")
        return [
            {
                "image_path": r["image_path"],
                "text": r["text"],
                "source": "persian-ocr-230k",
            }
            for r in rows
            if (r.get("text") or "").strip()
        ]

    script = project_root / "scripts" / "download_external_datasets.py"
    spec = importlib.util.spec_from_file_location("download_external_datasets", script)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print(f"[mix] downloading external persian-ocr-230k limit={limit}")
    records = mod.download_persian_ocr_230k(
        limit=limit if limit > 0 else 10000, output_dir=out_dir
    )
    save_jsonl(records, records_path)
    meta.write_text(
        json.dumps({"count": len(records), "source": "persian-ocr-230k"}, indent=2),
        encoding="utf-8",
    )
    return [
        {
            "image_path": r["image_path"],
            "text": r["text"],
            "source": "persian-ocr-230k",
        }
        for r in records
        if (r.get("text") or "").strip()
    ]


def _oversample_hard(gold: list[dict], *, factor: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    extra: list[dict] = []
    hard = [r for r in gold if _is_short(r["text"]) or _is_number_heavy(r["text"])]
    for _ in range(max(0, factor - 1)):
        for row in hard:
            clone = dict(row)
            clone["source"] = f"{row.get('source', 'gold')}_oversample"
            extra.append(clone)
    rng.shuffle(extra)
    print(f"[mix] oversample hard={len(hard)} extra={len(extra)} factor={factor}")
    return extra


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--lines-v3", type=Path, default=DEFAULT_LINES_V3)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--pseudo-cache", type=Path, default=DEFAULT_PSEUDO_CACHE)
    p.add_argument("--pseudo-limit", type=int, default=1500)
    p.add_argument("--external-limit", type=int, default=5000)
    p.add_argument("--synthetic-limit", type=int, default=2000)
    p.add_argument("--oversample-factor", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-pseudo", action="store_true")
    p.add_argument("--skip-external", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true")
    args = p.parse_args(argv)

    train_docs = _split_docs(args.split_dir, "train")
    val_docs = _split_docs(args.split_dir, "val")
    print(f"[mix] train_docs={sorted(train_docs)}")
    print(f"[mix] val_docs={sorted(val_docs)}")

    gold_train = _gold_rows(args.split_dir, "train")
    gold_val = _gold_rows(args.split_dir, "val")
    gold_paths = {r["image_path"] for r in gold_train}
    print(f"[mix] gold_train={len(gold_train)} gold_val={len(gold_val)}")

    mix: list[dict] = list(gold_train)
    mix.extend(_oversample_hard(gold_train, factor=args.oversample_factor, seed=args.seed))

    if not args.skip_pseudo:
        mix.extend(
            _pseudo_label_train_docs(
                lines_v3=args.lines_v3,
                train_docs=train_docs,
                gold_paths=gold_paths,
                cache_path=args.pseudo_cache,
                limit=args.pseudo_limit,
                project_root=PROJECT_ROOT,
            )
        )

    if not args.skip_external:
        external = _load_or_download_external(
            limit=args.external_limit, project_root=PROJECT_ROOT
        )
        mix.extend(external)
        if not external and not args.skip_synthetic:
            print("[mix] external empty — generating synthetic fallback")
            mix.extend(
                _generate_synthetic_lines(
                    limit=args.synthetic_limit,
                    out_dir=args.out_dir,
                    seed=args.seed,
                )
            )
    elif not args.skip_synthetic:
        mix.extend(
            _generate_synthetic_lines(
                limit=args.synthetic_limit,
                out_dir=args.out_dir,
                seed=args.seed,
            )
        )

    if not args.skip_synthetic and args.skip_external is False:
        # Always add a smaller synthetic number/punct set even when external works.
        mix.extend(
            _generate_synthetic_lines(
                limit=min(1000, args.synthetic_limit),
                out_dir=args.out_dir,
                seed=args.seed + 7,
            )
        )

    rng = random.Random(args.seed)
    rng.shuffle(mix)

    train_path = args.out_dir / "train_mix.jsonl"
    val_path = args.out_dir / "val_gold.jsonl"
    save_jsonl(mix, train_path)
    save_jsonl(gold_val, val_path)

    by_src: dict[str, int] = {}
    for r in mix:
        by_src[r.get("source", "?")] = by_src.get(r.get("source", "?"), 0) + 1
    summary = {
        "train_mix": str(train_path.relative_to(PROJECT_ROOT)),
        "val_gold": str(val_path.relative_to(PROJECT_ROOT)),
        "n_train": len(mix),
        "n_val_gold": len(gold_val),
        "by_source": by_src,
        "train_documents": sorted(train_docs),
        "val_documents": sorted(val_docs),
        "leakage_note": "pseudo/external never include val or gold_test documents",
    }
    summary_path = args.out_dir / "mix_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
