"""Run one engine over the truth lines and cache every prediction to disk.

    python -m htr_eval.run --engine surya --subset-pages 20
    python -m htr_eval.harness --pred htr_eval/cache/surya.jsonl --subset-pages 20

Predictions are appended to htr_eval/cache/<engine>.jsonl as they are produced; a
re-run skips lines already in the cache, so nothing is recognised twice.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from htr_eval.engines import load_engine
from htr_eval.harness import DEFAULT_TRUTH, ROOT, load_jsonl, load_truth, subset_pages

CACHE = ROOT / "htr_eval" / "cache"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine", required=True)
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    p.add_argument("--subset-pages", type=int, default=0)
    args = p.parse_args()

    rows = load_truth(args.truth)
    if args.subset_pages:
        rows = subset_pages(rows, args.subset_pages)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{args.engine}.jsonl"
    done = {r["image_path"] for r in load_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["image_path"] not in done]
    missing = [r["image_path"] for r in todo if not (ROOT / r["image_path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} line images not found, e.g. {missing[0]}")
    print(f"{args.engine}: {len(rows)} lines, {len(done & {r['image_path'] for r in rows})} cached, {len(todo)} to run")
    if not todo:
        return 0

    engine = load_engine(args.engine)
    t0 = time.perf_counter()
    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            s = time.perf_counter()
            res = engine.recognize(Image.open(ROOT / r["image_path"]).convert("RGB"))
            f.write(json.dumps({"image_path": r["image_path"], "page": r["page"], **res,
                                "seconds": round(time.perf_counter() - s, 3)}, ensure_ascii=False) + "\n")
            f.flush()
    n_pages = len({r["page"] for r in todo})
    print(f"done in {time.perf_counter() - t0:.1f}s ({(time.perf_counter() - t0) / n_pages:.1f}s/page)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
