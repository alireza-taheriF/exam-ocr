#!/usr/bin/env python3
"""Score a predictions jsonl against any gold-format jsonl.

Gold rows with empty text or illegible=true are not scored (they are counted).
Reports corpus CER/WER/char-accuracy overall, per source_document, and per crop
width bucket. Missing predictions count as empty output (CER 1.0 for that line).

    .venv/bin/python scripts/eval_ocr.py --gold dataset/dev_claude.jsonl \
        --pred dataset/preds/dev_base.jsonl --report dataset/reports/dev_base.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from pipeline.normalizer import normalize_text  # noqa: E402
import random  # noqa: E402

from training.eval.metrics import (  # noqa: E402
    accuracy_from_cer, cer, corpus_cer, corpus_wer, lenient_normalize, levenshtein)
from pipeline.normalizer import normalize_text as _strict  # noqa: E402

BUCKETS = [(0, 150, "<150"), (150, 250, "150-250"), (250, 500, "250-500"), (500, 10**9, ">=500")]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _dist_len(refs, hyps, norm):
    """Per-line (clamped distance, ref length) pairs, same rule as corpus_cer."""
    out = []
    for r, h in zip(refs, hyps):
        r, h = norm(r or ""), norm(h or "")
        if not r:
            out.append((len(h), len(h)))
        else:
            out.append((min(levenshtein(r, h), len(r)), len(r)))
    return out


def bootstrap_ci(pairs, n=1000, seed=0):
    rng = random.Random(seed)
    if not pairs:
        return None
    vals = []
    for _ in range(n):
        smp = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        d, l = sum(x[0] for x in smp), sum(x[1] for x in smp)
        vals.append(d / l if l else 0.0)
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[int(0.975 * n)], 4)]


def summarize(refs: list[str], hyps: list[str]) -> dict:
    if not refs:
        return {"lines": 0}
    c = corpus_cer(refs, hyps)
    cl = corpus_cer(refs, hyps, lenient=True)
    return {"lines": len(refs),
            "cer": round(c, 4), "cer_ci95": bootstrap_ci(_dist_len(refs, hyps, _strict)),
            "cer_lenient": round(cl, 4),
            "cer_lenient_ci95": bootstrap_ci(_dist_len(refs, hyps, lenient_normalize)),
            "wer": round(corpus_wer(refs, hyps), 4),
            "char_accuracy_pct": round(accuracy_from_cer(c), 2),
            "char_accuracy_lenient_pct": round(accuracy_from_cer(cl), 2),
            "exact_pct": round(100 * sum(cer(r, h) == 0 for r, h in zip(refs, hyps)) / len(refs), 2)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--min-width", type=int, default=0,
                   help="only score crops at least this wide (the rest are 'skipped')")
    a = p.parse_args()

    preds = {r["image_path"]: r["text"] for r in load(a.pred)}
    gold = load(a.gold)
    scored, illegible = [], 0
    for r in gold:
        if r.get("illegible") or not normalize_text(r.get("text") or ""):
            illegible += 1
            continue
        scored.append(r)
    for r in scored:
        r["_w"] = Image.open(ROOT / r["image_path"]).width
    kept = [r for r in scored if r["_w"] >= a.min_width]
    refs = [r["text"] for r in kept]
    hyps = [preds.get(r["image_path"], "") for r in kept]
    missing = sum(1 for r in kept if r["image_path"] not in preds)

    report = {"gold": str(a.gold), "pred": str(a.pred), "gold_rows": len(gold),
              "illegible_or_empty": illegible, "scored_before_width_filter": len(scored),
              "min_width": a.min_width, "skipped_by_width": len(scored) - len(kept),
              "missing_predictions": missing, "overall": summarize(refs, hyps)}
    by_doc = collections.defaultdict(lambda: ([], []))
    by_w = collections.defaultdict(lambda: ([], []))
    for r, h in zip(kept, hyps):
        by_doc[r["source_document"]][0].append(r["text"]); by_doc[r["source_document"]][1].append(h)
        for lo, hi, name in BUCKETS:
            if lo <= r["_w"] < hi:
                by_w[name][0].append(r["text"]); by_w[name][1].append(h)
    report["by_document"] = {d: summarize(*v) for d, v in sorted(by_doc.items())}
    report["by_width"] = {n: summarize(*by_w[n]) for _, _, n in BUCKETS if n in by_w}
    if a.min_width:  # also report on everything, counting skipped crops as empty
        allr = [r["text"] for r in scored]
        allh = [preds.get(r["image_path"], "") if r["_w"] >= a.min_width else "" for r in scored]
        report["overall_counting_skipped_as_empty"] = summarize(allr, allh)

    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
