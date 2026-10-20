"""Persistent local OCR worker for the FastAPI app.

The app's venv pins transformers<4.46 (surya 0.6) while the OCR engines need transformers>=5,
so recognition runs here, in a second interpreter (OCR_PYTHON). Protocol, one JSON per line:
stdout {"ready": true} once loaded; then stdin {"png": <base64>} -> stdout {"text": "..."}.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO

from PIL import Image

from htr_eval.engines import load_engine


def main() -> int:
    out, sys.stdout = sys.stdout, sys.stderr  # library prints must not corrupt the protocol
    engine = load_engine(os.environ.get("OCR_ENGINE", "qwen3vl-fa"))
    out.write('{"ready": true}\n')
    out.flush()
    for raw in sys.stdin:
        img = Image.open(BytesIO(base64.b64decode(json.loads(raw)["png"]))).convert("RGB")
        out.write(json.dumps({"text": engine.recognize(img)["text"]}, ensure_ascii=False) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
