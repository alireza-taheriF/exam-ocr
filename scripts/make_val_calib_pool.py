#!/usr/bin/env python3
"""Build the val pool for confidence-calibration labels, minus sealed crops.

`eval/new_gold` (branch cursor/report-accuracy-graphs) was sampled from all
documents, so 108 of its 300 sealed crops sit in val_v3. Calibration labels
must not touch them, nor the current gold. Writes dataset/val_calib_pool.jsonl;
label from it with:

    .venv/bin/python scripts/label_gold_interactive.py \
        --source dataset/val_calib_pool.jsonl --out dataset/val_calib.jsonl --target 150
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.sealed import sealed_crop_names  # noqa: E402


def main() -> int:
    sealed = sealed_crop_names()
    rows = [json.loads(l) for l in (PROJECT_ROOT / "dataset" / "val_v3.jsonl").open(encoding="utf-8") if l.strip()]
    pool = [r for r in rows if Path(r["image_path"]).name not in sealed]
    out = PROJECT_ROOT / "dataset" / "val_calib_pool.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pool), encoding="utf-8")
    print(f"val {len(rows)} -> pool {len(pool)} (dropped {len(rows) - len(pool)} sealed); wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
