#!/usr/bin/env python3
"""
Download and prepare external Persian OCR datasets.

Usage:
    python scripts/download_external_datasets.py --dataset persian-ocr-230k --limit 5000
    python scripts/download_external_datasets.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "dataset" / "external"
ALL_TRAIN = PROJECT_ROOT / "dataset" / "all_train.jsonl"

DATASETS = {
    "persian-ocr-230k": {
        "hf_id": "ordaktaktak/Persian-OCR-230k",
        "description": "Persian printed text (~230k samples)",
    },
}


def list_datasets() -> None:
    for name, info in DATASETS.items():
        print(f"  {name}: {info['description']} ({info['hf_id']})")


def download_persian_ocr_230k(*, limit: int | None, output_dir: Path) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install: pip install datasets")

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Loading {DATASETS['persian-ocr-230k']['hf_id']} ...")
    ds = load_dataset(DATASETS["persian-ocr-230k"]["hf_id"], split="train", streaming=True)

    records: list[dict] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break

        image = row.get("image")
        text = row.get("text") or row.get("label") or ""
        if image is None or not str(text).strip():
            continue

        fname = f"pocr_{i:06d}.jpg"
        img_path = images_dir / fname
        image.convert("RGB").save(img_path, quality=90)

        records.append({
            "image_path": str(img_path.relative_to(PROJECT_ROOT)),
            "text": str(text).strip(),
            "source": "persian-ocr-230k",
        })

        if (i + 1) % 500 == 0:
            print(f"  downloaded {i + 1}...")

    meta = output_dir / "meta.json"
    meta.write_text(json.dumps({"count": len(records), "source": "persian-ocr-230k"}, indent=2))
    return records


def merge_all_train(extra_records: list[dict]) -> int:
    existing: list[dict] = []
    if ALL_TRAIN.exists():
        with ALL_TRAIN.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing.append(json.loads(line))

    seen = {r.get("image_path") for r in existing}
    added = 0
    for rec in extra_records:
        if rec["image_path"] not in seen:
            existing.append(rec)
            seen.add(rec["image_path"])
            added += 1

    with ALL_TRAIN.open("w", encoding="utf-8") as f:
        for rec in existing:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Download external Persian OCR datasets")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()))
    parser.add_argument("--limit", type=int, default=1000, help="Max samples (default 1000 for trial)")
    parser.add_argument("--merge", action="store_true", help="Merge into dataset/all_train.jsonl")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        list_datasets()
        return 0

    if not args.dataset:
        parser.print_help()
        return 1

    out_dir = EXTERNAL_DIR / args.dataset
    if args.dataset == "persian-ocr-230k":
        records = download_persian_ocr_230k(limit=args.limit, output_dir=out_dir)
    else:
        print(f"Not implemented: {args.dataset}")
        return 1

    print(f"Downloaded {len(records)} samples → {out_dir}")

    if args.merge and records:
        added = merge_all_train(records)
        print(f"Merged {added} new records into {ALL_TRAIN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
