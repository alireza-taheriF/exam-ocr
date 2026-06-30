#!/usr/bin/env python3
"""
Smoke-test a TrOCR export before copying to exam-ocr/models/.

Usage (local, after downloading from Colab):
    python training/colab/export_for_inference.py models/trocr-fa-exam-v1
    python training/colab/export_for_inference.py models/trocr-fa-exam-v1 dataset/lines/0000a450c8bb.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Test exported TrOCR model")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("image", type=Path, nargs="?", default=None)
    args = parser.parse_args()

    if not args.model_dir.exists():
        print(f"Model not found: {args.model_dir}")
        return 1

    try:
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("Install: pip install transformers torch pillow")
        return 1

    processor = TrOCRProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    model.eval()

    if args.image is None:
        lines_dir = PROJECT_ROOT / "dataset" / "lines"
        images = sorted(lines_dir.glob("*.jpg"))
        if not images:
            print("No sample image; pass a path as second argument.")
            return 1
        args.image = images[0]

    image = Image.open(args.image).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    generated = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(f"Image: {args.image}")
    print(f"OCR:   {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
