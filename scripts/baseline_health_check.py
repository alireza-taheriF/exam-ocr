#!/usr/bin/env python3
"""
Health-check the Exam OCR pipeline on sample images.

Runs layout detection (+ optional OCR) and writes a report to logs/baseline_report.txt.

Usage:
    python scripts/baseline_health_check.py
    python scripts/baseline_health_check.py --with-ocr
    python scripts/baseline_health_check.py --count 5
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

RAW_DIR = PROJECT_ROOT / "dataset" / "raw_images"
REPORT_PATH = PROJECT_ROOT / "logs" / "baseline_report.txt"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def _vlm_config_status() -> str:
    key = os.environ.get("VLM_API_KEY", "").strip()
    base = os.environ.get("VLM_API_BASE", "(not set)")
    model = os.environ.get("VLM_MODEL", "(not set)")
    placeholders = {"", "your_api_key_here", "changeme", "replace_me"}
    if not key or key.lower() in placeholders:
        return f"NOT CONFIGURED (base={base}, model={model})"
    return f"configured (base={base}, model={model})"


def run_health_check(*, sample_count: int, with_ocr: bool) -> str:
    from pipeline.layout import detect_blocks

    if with_ocr:
        from pipeline.ocr import run_ocr

    images = sorted(RAW_DIR.iterdir())[:sample_count]
    if not images:
        raise SystemExit(f"No images found in {RAW_DIR}")

    lines: list[str] = []
    lines.append("Exam OCR Baseline Health Check")
    lines.append(f"Date: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"VLM: {_vlm_config_status()}")
    lines.append(f"Samples: {sample_count} | OCR: {'yes' if with_ocr else 'layout only'}")
    lines.append("")

    ok = 0
    for path in images:
        lines.append(f"Image: {path.name}")
        try:
            rgb = _load_rgb(path)
            h, w = rgb.shape[:2]
            lines.append(f"  size: {w}x{h}")

            blocks = detect_blocks(rgb)
            lines.append(f"  layout blocks: {len(blocks)}")
            type_counts: dict[str, int] = {}
            for b in blocks:
                t = str(b.get("type", "unknown"))
                type_counts[t] = type_counts.get(t, 0) + 1
            lines.append(f"  block types: {type_counts}")

            if with_ocr:
                blocks = run_ocr(rgb, blocks)
                text_n = sum(1 for b in blocks if b.get("text") and str(b["text"]).strip())
                crop_n = sum(1 for b in blocks if b.get("crop_image") is not None)
                illeg_n = sum(1 for b in blocks if b.get("illegible"))
                lines.append(f"  OCR text blocks: {text_n}")
                lines.append(f"  OCR cropped: {crop_n}")
                lines.append(f"  OCR illegible: {illeg_n}")

            lines.append("  status: OK")
            ok += 1
        except Exception as exc:
            lines.append(f"  status: FAILED — {exc}")

        lines.append("")

    lines.append(f"Summary: {ok}/{len(images)} images passed")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline health check")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--with-ocr", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    report = run_health_check(sample_count=args.count, with_ocr=args.with_ocr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
