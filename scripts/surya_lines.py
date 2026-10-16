#!/usr/bin/env python3
"""Detect text-line boxes with Surya and print them as JSON.

Surya 0.6 needs transformers 4.45, the Qwen engine needs transformers 5, so
Surya lives in its own environment (.venv-surya) and pipeline/segment.py calls
this script as a subprocess (SEGMENTER=surya_cli).

    .venv-surya/bin/python scripts/surya_lines.py page.png   # → [[x1,y1,x2,y2], ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub"))

from PIL import Image  # noqa: E402
from surya.detection import batch_text_detection  # noqa: E402
from surya.model.detection.model import load_model, load_processor  # noqa: E402


def main() -> int:
    image = Image.open(sys.argv[1]).convert("RGB")
    result = batch_text_detection([image], load_model(), load_processor())[0]
    boxes = [[int(round(v)) for v in b.bbox] for b in result.bboxes]
    print(json.dumps(boxes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
