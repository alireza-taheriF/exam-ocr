#!/usr/bin/env python3
"""Run one OCR engine on a few gold crops and print text, confidence and time.

A quick "does the real engine load and produce sane output on this machine"
check — not an accuracy measurement (use scripts/eval_gate.py for that).

    HF_HUB_OFFLINE=1 QWEN_DEVICE=mps .venv-qwen/bin/python scripts/smoke_engine.py --n 5
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

from PIL import Image  # noqa: E402

from pipeline.engines import get_engine  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--engine", default="qwen_local")
    p.add_argument("--gold", type=Path, default=PROJECT_ROOT / "dataset" / "gold.jsonl")
    p.add_argument("--n", type=int, default=5)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.gold.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if (r.get("text") or "").strip()][: args.n]

    t0 = time.perf_counter()
    engine = get_engine(args.engine)
    device = getattr(getattr(engine, "model", None), "device", "n/a")
    print(f"engine={engine.version} device={device} load={time.perf_counter() - t0:.1f}s")

    for row in rows:
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        t = time.perf_counter()
        text, conf = engine.recognize(image)
        dt = time.perf_counter() - t
        conf_s = "None" if conf is None else f"{conf:.3f}"
        print(f"{dt:5.2f}s conf={conf_s} {image.width}x{image.height}")
        print(f"   gold: {row['text']}")
        print(f"   pred: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
