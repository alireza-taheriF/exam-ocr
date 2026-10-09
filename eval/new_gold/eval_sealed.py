#!/usr/bin/env python3
"""Evaluate any model predictions against the sealed new_gold corrections.

Requires ``corrections.csv`` with non-empty ``corrected_text`` rows.
Predictions JSONL: ``image_path``/``crop_path`` + ``text``/``pred``.

Usage:
  python -m eval.new_gold.eval_sealed --preds path/to/preds.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (  # noqa: E402
    bootstrap_mean_ci,
    cer,
    sentence_similarity,
    wer,
)
from eval.normalize import normalize_text  # noqa: E402

NEW_GOLD_DIR = Path(__file__).resolve().parent


def load_corrections(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            corrected = (r.get("corrected_text") or "").strip()
            if not corrected:
                continue
            rows.append(
                {
                    "line_id": r.get("line_id") or "",
                    "crop_path": r.get("crop_path") or r.get("image_path") or "",
                    "ref": corrected,
                }
            )
    return rows


def load_preds(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            ip = r.get("image_path") or r.get("crop_path") or ""
            if ip:
                out[ip] = r.get("text") or r.get("pred") or ""
    return out


def evaluate(corrections: list[dict], preds: dict[str, str], *, n_boot: int = 1000) -> dict:
    sims: list[float] = []
    refs: list[str] = []
    hyps: list[str] = []
    missing = 0
    for row in corrections:
        ip = row["crop_path"]
        if ip not in preds:
            missing += 1
            continue
        ref = row["ref"]
        pred = preds[ip]
        sims.append(sentence_similarity(pred, ref))
        refs.append(normalize_text(ref))
        hyps.append(normalize_text(pred))
    if not sims:
        return {
            "n_labeled": len(corrections),
            "n_scored": 0,
            "missing_preds": missing,
            "error": "No scored pairs — fill corrected_text and provide preds.",
        }
    ref_join = "\n".join(refs)
    hyp_join = "\n".join(hyps)
    boot = bootstrap_mean_ci(sims, n_boot=n_boot)
    high = sum(1 for s in sims if s >= 0.80) / len(sims)
    return {
        "n_labeled": len(corrections),
        "n_scored": len(sims),
        "missing_preds": missing,
        "mean_sentence_similarity": round(sum(sims) / len(sims), 6),
        "bootstrap_sentence_similarity": boot,
        "pct_similarity_ge_0_80": round(high, 6),
        "cer": round(cer(ref_join, hyp_join), 6),
        "wer": round(wer(ref_join, hyp_join), 6),
        "sealed": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corrections", type=Path, default=NEW_GOLD_DIR / "corrections.csv")
    p.add_argument("--preds", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval" / "phase_c_verify" / "new_gold_metrics.json",
    )
    args = p.parse_args(argv)

    seal = NEW_GOLD_DIR / "SEALED.json"
    if seal.exists():
        meta = json.loads(seal.read_text(encoding="utf-8"))
        if not meta.get("hash_check_ok", True):
            print("WARNING: SEALED.json reports hash_check_ok=false")

    corrections = load_corrections(args.corrections)
    if not corrections:
        print(
            f"No non-empty corrected_text in {args.corrections}. "
            "Fill the correction template first."
        )
        return 2
    preds = load_preds(args.preds)
    report = evaluate(corrections, preds, n_boot=args.n_boot)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
