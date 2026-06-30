#!/usr/bin/env python3
"""
Remove hallucinated/invalid labels from dataset.jsonl.

Writes cleaned file and a report of removed entries.

Usage:
    python scripts/clean_labels.py
    python scripts/clean_labels.py --inplace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.eval.label_quality import is_valid_label, rejection_reason  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "dataset.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "dataset_clean.jsonl"
REJECTED_LOG = PROJECT_ROOT / "dataset" / "labels_rejected.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean hallucinated labels")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 1

    output = args.input if args.inplace else args.output
    kept: list[dict] = []
    rejected: list[dict] = []

    with args.input.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                rejected.append({"line": line_no, "error": "invalid_json"})
                continue

            text = str(record.get("text") or "")
            if is_valid_label(text):
                kept.append(record)
            else:
                rejected.append({
                    "image_path": record.get("image_path"),
                    "reason": rejection_reason(text),
                    "text_preview": text[:120],
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with REJECTED_LOG.open("w", encoding="utf-8") as f:
        for record in rejected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Kept: {len(kept)} | Rejected: {len(rejected)} | Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
