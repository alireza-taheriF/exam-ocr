#!/usr/bin/env python3
"""Pick the lines to hand-label: stratified by document, never from a gold document.

    python scripts/build_label_queue.py            # writes dataset/label_queue.jsonl (600 lines)

train_v3.jsonl has only 3 documents (cap 150 each = 450), so the pool is every line in lines_v3.jsonl
(geometry filters already applied by extract_and_align_lines.py) minus gold documents, minus web photos,
minus documents that are not Persian prose (English notes, math/equation sheets).
Fragment proxy: crop at least 250x32 px with aspect 4..16 (the repo has no fragment classifier).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKIP_DOCS = {"Summary_of_Yule", "تحلیل_سازه_مهندس_طاحونی"}


def load(p):
    return [json.loads(l) for l in (ROOT / p).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=600)
    ap.add_argument("--max-per-doc", type=int, default=150)
    args = ap.parse_args()
    gold_docs = {r["source_document"] for r in load("dataset/gold.jsonl")}
    by = defaultdict(list)
    for r in load("dataset/lines_v3.jsonl"):
        d = r["source_document"]
        if d in gold_docs or d in SKIP_DOCS or d.startswith("web/"):
            continue
        w, h = Image.open(ROOT / r["image_path"]).size
        if w >= 250 and h >= 32 and 4 <= w / h <= 16:
            by[d].append(r)
    docs = sorted(d for d, v in by.items() if len(v) >= 100)
    quota = min(args.max_per_doc, -(-args.total // len(docs)))
    rng = random.Random(0)
    rows = [r for d in docs for r in rng.sample(by[d], min(quota, len(by[d])))][: args.total]
    rng.shuffle(rows)  # mix documents so the session does not feel monotone
    out = ROOT / "dataset" / "label_queue.jsonl"
    out.write_text("".join(json.dumps({k: r[k] for k in ("image_path", "source_document", "source_image", "line_index")}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert not {r["source_document"] for r in rows} & gold_docs
    print(f"{len(rows)} lines from {len(docs)} documents (quota {quota}) -> {out.name}")


if __name__ == "__main__":
    main()
