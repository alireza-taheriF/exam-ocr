#!/usr/bin/env python3
"""Sample a stratified dev set from dataset/val_v3.jsonl.

Dev is where every tuning decision is made; gold.jsonl stays untouched. Only
documents with at least --min-lines lines are eligible, the budget is spread
evenly across those documents, and inside each document the crops are drawn so
their width distribution follows the whole corpus (deciles of train+val+test).

    .venv/bin/python scripts/build_dev_set.py            # -> dataset/dev_todo.jsonl
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def width(row: dict) -> int:
    b = row.get("crop_bbox") or row["bbox"]
    return int(b[2] - b[0])


def build(val, corpus_widths, forbidden, n, min_lines, seed):
    rng = random.Random(seed)
    edges = statistics.quantiles(corpus_widths, n=10)
    bin_of = lambda r: bisect.bisect(edges, width(r))  # noqa: E731

    by_doc = collections.defaultdict(list)
    for r in val:
        if r["image_path"] not in forbidden:
            by_doc[r["source_document"]].append(r)
    docs = sorted(d for d, rs in by_doc.items() if len(rs) >= min_lines)
    if not docs:
        raise SystemExit("no eligible documents")

    # Even split, capped by what each document has; the shortfall of small
    # documents is redistributed over the larger ones.
    quota = {d: 0 for d in docs}
    open_docs = list(docs)
    left = n
    while left > 0 and open_docs:
        share, extra = divmod(left, len(open_docs))
        for i, d in enumerate(list(open_docs)):
            add = min(share + (1 if i < extra else 0), len(by_doc[d]) - quota[d])
            quota[d] += add
            left -= add
            if quota[d] >= len(by_doc[d]):
                open_docs.remove(d)

    picked = []
    for d in docs:
        bins = collections.defaultdict(list)
        for r in by_doc[d]:
            bins[bin_of(r)].append(r)
        for b in bins.values():
            rng.shuffle(b)
        want = {b: round(quota[d] / 10) for b in range(10)}
        got = []
        for b in range(10):  # each decile gets its share of the quota
            got += bins[b][: want[b]]
        pool = [r for b in range(10) for r in bins[b][want[b]:]]
        rng.shuffle(pool)
        got += pool[: quota[d] - len(got)]
        got = got[: quota[d]]
        picked += got
    rng.shuffle(picked)
    return picked, docs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--min-lines", type=int, default=20)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--out", type=Path, default=DS / "dev_todo.jsonl")
    a = p.parse_args()

    val = load(DS / "val_v3.jsonl")
    corpus = val + load(DS / "train_v3.jsonl") + load(DS / "test_v3.jsonl")
    gold = {r["image_path"] for r in load(DS / "gold.jsonl")}
    picked, docs = build(val, [width(r) for r in corpus], gold, a.n, a.min_lines, a.seed)

    with a.out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps({
                "image_path": r["image_path"],
                "source_document": r["source_document"],
                "width": width(r),
            }, ensure_ascii=False) + "\n")
    per = collections.Counter(r["source_document"] for r in picked)
    print(f"wrote {len(picked)} rows to {a.out}")
    for d in docs:
        print(f"  {per[d]:4d}  {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
