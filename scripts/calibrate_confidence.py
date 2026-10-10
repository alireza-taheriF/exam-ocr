#!/usr/bin/env python3
"""Relate engine confidence to CER on hand-labeled VAL lines and suggest
LOW_CONFIDENCE_THRESHOLD (the UI highlight). Never run this on gold.

    .venv/bin/python scripts/label_gold_interactive.py \
        --source dataset/val_calib_pool.jsonl --out dataset/val_calib.jsonl --target 150
    HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/calibrate_confidence.py

Writes dataset/reports/calibration_val.json: CER per confidence decile, and
for each candidate threshold the share of lines flagged, the share of
high-error lines (CER > 0.3) caught (recall) and the share of flagged lines
that really were bad (precision). The suggestion is the lowest threshold that
catches ≥ 80% of the high-error lines.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import cer  # noqa: E402
from training.sealed import sealed_crop_names  # noqa: E402

BAD_CER = 0.3
TARGET_RECALL = 0.8


def table(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["confidence"])
    deciles = []
    for i in range(10):
        chunk = rows[i * len(rows) // 10:(i + 1) * len(rows) // 10]
        if chunk:
            deciles.append({"conf_min": round(chunk[0]["confidence"], 3), "conf_max": round(chunk[-1]["confidence"], 3),
                            "lines": len(chunk), "mean_line_cer": round(sum(r["cer"] for r in chunk) / len(chunk), 3)})
    bad = [r for r in rows if r["cer"] > BAD_CER]
    thresholds = []
    for t in [x / 100 for x in range(30, 96, 5)]:
        flagged = [r for r in rows if r["confidence"] < t]
        caught = [r for r in flagged if r["cer"] > BAD_CER]
        thresholds.append({"threshold": t, "flagged_pct": round(100 * len(flagged) / len(rows), 1),
                           "bad_recall_pct": round(100 * len(caught) / len(bad), 1) if bad else None,
                           "flag_precision_pct": round(100 * len(caught) / len(flagged), 1) if flagged else None})
    ok = [t for t in thresholds if t["bad_recall_pct"] is not None and t["bad_recall_pct"] >= 100 * TARGET_RECALL]
    return {"lines": len(rows), "bad_lines": len(bad), "deciles": deciles, "thresholds": thresholds,
            "suggested_threshold": ok[0]["threshold"] if ok else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", type=Path, default=PROJECT_ROOT / "dataset" / "val_calib.jsonl")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "dataset" / "reports" / "calibration_val.json")
    args = p.parse_args()

    sealed = sealed_crop_names()
    labels = [json.loads(l) for l in args.labels.open(encoding="utf-8") if l.strip()]
    if any(Path(r["image_path"]).name in sealed for r in labels):
        sys.exit("calibration labels contain sealed (gold/test/new_gold) crops")
    labels = [r for r in labels if normalize_text(r.get("text") or "")]
    print(f"{len(labels)} legible labeled val lines")

    from PIL import Image

    from pipeline.engines import QwenLocalEngine

    engine = QwenLocalEngine()
    rows = []
    for r in labels:
        image = Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB")
        t0 = time.perf_counter()
        text, conf = engine.recognize(image)
        rows.append({"image_path": r["image_path"], "gold": r["text"], "pred": text,
                     "confidence": conf if conf is not None else 0.0, "cer": cer(r["text"], text),
                     "seconds": round(time.perf_counter() - t0, 2)})
    result = {"engine": engine.version, **table(rows), "rows": rows}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in result["thresholds"]:
        print(t)
    print(f"suggested LOW_CONFIDENCE_THRESHOLD = {result['suggested_threshold']}  (wrote {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
