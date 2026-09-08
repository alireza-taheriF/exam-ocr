#!/usr/bin/env python3
"""Score OCR predictions against a hand-verified gold subset.

Step 0 found no hand-verified gold set in this repo, and that is the single
blocking gap for measuring accuracy. Every label in `dataset/` is *pseudo-label*
output from an auto-labeler, so CER computed against it measures agreement with
that labeler, not accuracy. This script exists to make a real gold set pay off
the moment it exists.

## Creating the gold set (≈2 hours, and worth more than any other 2 hours here)

Run the interactive labeler — it implements the sampling described below, shows
each crop, and appends every entry as you type it so an interrupt costs at most
one line:

    python scripts/label_gold_interactive.py            # 240 lines, resumable
    python scripts/label_gold_interactive.py --limit 40 # a short session

It draws from `dataset/test_v3.jsonl` (the held-out split, so gold never informs
training or backend selection) and reuses the manifest's `image_path` values,
which is what lets predictions join to gold at all. Sampling is restricted to
documents with ≥50 lines: test has 60 groups but 56 are single-page crawled web
images contributing 1-3 lines each, so an even draw across all 60 would spend the
typing budget on web screenshots rather than exam handwriting.

Then score any prediction set against it:

    python training/eval/eval_gold_subset.py --gold dataset/gold.jsonl \
        --pred predictions.jsonl --by-document

Both files are JSONL keyed on `image_path`; predictions need only
`{"image_path": ..., "text": ...}`. Rows present in gold but missing from
predictions are counted as empty hypotheses, not skipped — a model that declines
to predict must not score better than one that predicts badly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import (  # noqa: E402
    accuracy_from_cer,
    cer,
    corpus_cer,
    corpus_wer,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Corpus CER/WER against a gold subset")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--by-document", action="store_true",
                   help="Also break results down per source_document")
    p.add_argument("--worst", type=int, default=10,
                   help="Show the N worst lines (0 to disable)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    for f in (args.gold, args.pred):
        if not f.exists():
            print(f"Not found: {f}")
            return 1

    gold_rows = [r for r in load_jsonl(args.gold) if normalize_text(r.get("text") or "")]
    if not gold_rows:
        print(f"{args.gold} has no non-empty labels.")
        print("A gold set must be hand-transcribed — see this file's docstring.")
        return 1

    pred = {r["image_path"]: (r.get("text") or "") for r in load_jsonl(args.pred)}

    refs, hyps, missing, per_doc = [], [], 0, defaultdict(lambda: ([], []))
    for row in gold_rows:
        key = row["image_path"]
        hyp = pred.get(key)
        if hyp is None:
            missing += 1
            hyp = ""
        refs.append(row["text"])
        hyps.append(hyp)
        doc = row.get("source_document", "unknown")
        per_doc[doc][0].append(row["text"])
        per_doc[doc][1].append(hyp)

    result = {
        "gold": str(args.gold),
        "pred": str(args.pred),
        "lines_scored": len(refs),
        "predictions_missing": missing,
        "documents": len(per_doc),
        "corpus_cer": round(corpus_cer(refs, hyps), 4),
        "corpus_wer": round(corpus_wer(refs, hyps), 4),
    }
    result["char_accuracy_pct"] = round(accuracy_from_cer(result["corpus_cer"]), 2)
    result["exact_match_pct"] = round(
        100 * sum(cer(r, h) == 0.0 for r, h in zip(refs, hyps)) / len(refs), 2
    )

    print(f"\n  gold lines scored   : {result['lines_scored']} "
          f"across {result['documents']} document(s)")
    if missing:
        print(f"  MISSING predictions : {missing} (scored as empty hypotheses)")
    print(f"  corpus CER          : {result['corpus_cer']:.4f}")
    print(f"  corpus WER          : {result['corpus_wer']:.4f}")
    print(f"  char accuracy       : {result['char_accuracy_pct']:.2f}%")
    print(f"  exact-line match    : {result['exact_match_pct']:.2f}%")

    if args.by_document:
        print("\n  per document:")
        rows = sorted(
            ((d, len(r), corpus_cer(r, h)) for d, (r, h) in per_doc.items()),
            key=lambda t: -t[2],
        )
        result["by_document"] = [
            {"document": d, "lines": n, "corpus_cer": round(c, 4)} for d, n, c in rows
        ]
        for d, n, c in rows:
            print(f"    CER {c:6.4f}  n={n:4d}  {d}")

    if args.worst:
        print(f"\n  worst {args.worst} lines:")
        worst = sorted(
            zip(refs, hyps, (cer(r, h) for r, h in zip(refs, hyps))),
            key=lambda t: -t[2],
        )[: args.worst]
        for ref, hyp, c in worst:
            print(f"    CER {c:6.4f}\n      ref: {ref[:90]}\n      hyp: {hyp[:90]}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
