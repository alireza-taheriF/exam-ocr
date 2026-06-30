#!/usr/bin/env python3
"""
Filter low-quality images from dataset/raw_images.

Removes images that are too small, too blurry, or corrupt.
Moves rejected files to dataset/rejected/ and logs reasons.

Usage:
    python scripts/filter_dataset.py
    python scripts/filter_dataset.py --min-size 400 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "dataset" / "raw_images"
REJECTED_DIR = PROJECT_ROOT / "dataset" / "rejected"
LOG_PATH = PROJECT_ROOT / "dataset" / "filter_log.jsonl"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _load_grayscale(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        from PIL import Image

        return np.array(Image.open(path).convert("L"))
    except Exception:
        return None


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def evaluate_image(path: Path, *, min_size: int, min_blur: float) -> str | None:
    gray = _load_grayscale(path)
    if gray is None:
        return "corrupt"

    h, w = gray.shape[:2]
    if min(w, h) < min_size:
        return f"too_small_{w}x{h}"

    if blur_score(gray) < min_blur:
        return "too_blurry"

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter dataset images by quality")
    parser.add_argument("--input-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--min-size", type=int, default=400)
    parser.add_argument("--min-blur", type=float, default=50.0, help="Laplacian variance threshold")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Input dir not found: {args.input_dir}")
        return 1

    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0

    for path in sorted(args.input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        reason = evaluate_image(path, min_size=args.min_size, min_blur=args.min_blur)
        if reason is None:
            kept += 1
            continue

        rejected += 1
        record = {"path": path.name, "reason": reason}
        print(f"REJECT {path.name}: {reason}")

        if not args.dry_run:
            dest = REJECTED_DIR / path.name
            shutil.move(str(path), str(dest))
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone: kept={kept} rejected={rejected} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
