#!/usr/bin/env python3
"""Build the crop pool Claude labels for LoRA training (train + val splits).

Never includes sealed crops (training/sealed.py) or dataset/val_calib.jsonl,
which stays the calibration set. Big documents are capped so the 3 train
notebooks don't drown the 58 val documents, and a cheap geometry filter drops
crops that are almost never text (spiral rings, page-sized blocks, blanks):
on the 388 hand/Claude-labeled crops it kept 212/218 legible lines and
dropped 65/170 junk ones.

Rows are interleaved round-robin across documents, so labeling the sheets in
order and stopping at any point still gives a document-balanced set.

    .venv/bin/python scripts/make_claude_label_pool.py
    .venv/bin/python scripts/claude_label_sheets.py make dataset/claude_label_pool.jsonl outputs/claude_label/pool
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.sealed import sealed_crop_names  # noqa: E402

MIN_ASPECT = 1.5
MIN_INK = 0.02  # share of pixels clearly darker than the crop's mean


def looks_like_text(path: Path) -> bool:
    im = Image.open(path).convert("L")
    if im.width / im.height < MIN_ASPECT:
        return False
    a = np.asarray(im, dtype=float)
    return (a < a.mean() - 40).mean() >= MIN_INK


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cap-train", type=int, default=900, help="max crops sampled per train document")
    p.add_argument("--cap-val", type=int, default=450, help="max crops sampled per val document")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset" / "claude_label_pool.jsonl")
    args = p.parse_args()

    excluded = sealed_crop_names()
    calib = PROJECT_ROOT / "dataset" / "val_calib.jsonl"
    excluded |= {Path(json.loads(l)["image_path"]).name for l in calib.open(encoding="utf-8") if l.strip()}

    rnd = random.Random(args.seed)
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for split, cap in (("train", args.cap_train), ("val", args.cap_val)):
        docs: dict[str, list[dict]] = defaultdict(list)
        for line in (PROJECT_ROOT / "dataset" / f"{split}_v3.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            if Path(r["image_path"]).name not in excluded:
                docs[r["source_document"]].append({**r, "split": split})
        for doc, rows in docs.items():
            rnd.shuffle(rows)
            by_doc[doc] = [r for r in rows[:cap] if looks_like_text(PROJECT_ROOT / r["image_path"])]

    queues = sorted(by_doc.values(), key=len)
    pool = []
    while any(queues):
        for q in queues:
            if q:
                pool.append(q.pop())
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8")
    per_split = {s: sum(r["split"] == s for r in pool) for s in ("train", "val")}
    print(f"{len(pool)} crops from {len(by_doc)} documents {per_split} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
