# DEPRECATED — see scripts/split_dataset.py (grouped)
#
# Kept as reference only. split_rows() shuffles flat line rows with no grouping
# by source page or document, so sibling lines from the same scanned page land in
# different splits. Measured leakage: 876/878 val pages (99.8%) and 865/869 test
# pages (99.5%) also appear in train; all 12 source documents appear in all three
# splits. The docstring's 'test set is hold-out' claim was never true.
# Do not run this script.
#!/usr/bin/env python3
"""
Split lines.jsonl into train/val/test sets (70/15/15).

The test set is hold-out — never use for training.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LINES_JSONL = PROJECT_ROOT / "dataset" / "lines.jsonl"
TRAIN_OUT = PROJECT_ROOT / "dataset" / "train.jsonl"
VAL_OUT = PROJECT_ROOT / "dataset" / "val.jsonl"
TEST_OUT = PROJECT_ROOT / "dataset" / "test.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict], *, seed: int, train_ratio: float, val_ratio: float) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=Path, default=LINES_JSONL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        print("Run: python scripts/extract_lines.py first")
        return 1

    rows = load_jsonl(args.input)
    if not rows:
        print("No rows in input")
        return 1

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        print("Invalid ratios")
        return 1

    train, val, test = split_rows(
        rows, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)
    write_jsonl(TEST_OUT, test)

    print(f"Total: {len(rows)}")
    print(f"  train: {len(train)} → {TRAIN_OUT}")
    print(f"  val:   {len(val)} → {VAL_OUT}")
    print(f"  test:  {len(test)} → {TEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
