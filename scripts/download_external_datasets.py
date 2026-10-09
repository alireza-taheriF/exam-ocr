#!/usr/bin/env python3
"""
Download and prepare external Persian OCR datasets.

Usage:
    python scripts/download_external_datasets.py --dataset persian-ocr-230k --limit 5000
    python scripts/download_external_datasets.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = PROJECT_ROOT / "dataset" / "external"
ALL_TRAIN = PROJECT_ROOT / "dataset" / "all_train.jsonl"

DATASETS = {
    "persian-ocr-230k": {
        "hf_id": "ordaktaktak/Persian-OCR-230k",
        "description": "Persian text line images (~230k; Images.zip + CSV)",
    },
}


def list_datasets() -> None:
    for name, info in DATASETS.items():
        print(f"  {name}: {info['description']} ({info['hf_id']})")


def download_persian_ocr_230k(*, limit: int | None, output_dir: Path) -> list[dict]:
    """Download Images.zip + train.csv and materialize local JPEG crops."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    repo = DATASETS["persian-ocr-230k"]["hf_id"]
    print(f"Downloading {repo} (Images.zip + train.csv) ...")
    zip_path = Path(
        hf_hub_download(repo, "data/Images.zip", repo_type="dataset")
    )
    csv_path = Path(
        hf_hub_download(repo, "data/train.csv", repo_type="dataset")
    )

    # Map fname -> text from CSV (columns: fname,text or similar).
    labels: dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("fname") or row.get("file") or "").strip()
            text = (row.get("text") or row.get("label") or "").strip()
            if fname and text:
                # Normalize key to basename for zip lookup.
                labels[fname] = text
                labels[Path(fname).name] = text

    records: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".jpg", ".jpeg", ".png"))]
        for i, member in enumerate(members):
            if limit is not None and len(records) >= limit:
                break
            base = Path(member).name
            # CSV may store Images/img_x.jpg
            text = labels.get(member) or labels.get(f"Images/{base}") or labels.get(base)
            if not text:
                continue
            out_name = f"pocr_{len(records):06d}{Path(base).suffix.lower() or '.jpg'}"
            out_path = images_dir / out_name
            if not out_path.exists():
                with zf.open(member) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
            records.append(
                {
                    "image_path": str(out_path.relative_to(PROJECT_ROOT)),
                    "text": text,
                    "source": "persian-ocr-230k",
                }
            )
            if (len(records)) % 500 == 0:
                print(f"  extracted {len(records)}...")

    meta = output_dir / "meta.json"
    meta.write_text(
        json.dumps({"count": len(records), "source": "persian-ocr-230k"}, indent=2),
        encoding="utf-8",
    )
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

    records_path = out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Downloaded {len(records)} samples → {out_dir}")

    if args.merge and records:
        added = merge_all_train(records)
        print(f"Merged {added} new records into {ALL_TRAIN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
