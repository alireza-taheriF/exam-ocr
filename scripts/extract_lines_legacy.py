# DEPRECATED — see scripts/extract_and_align_lines.py
#
# Kept as reference only. The alignment strategy here is unsound: line 96 does
#   count = min(len(bboxes), len(label_lines))
# and pairs label_lines[i] with bboxes[i] by list index. Surya detects a median
# of 31 boxes/page while the page-level VLM emitted a median of 17 newline-
# separated lines, so after the first divergence every later pairing is shifted.
# Measured result: 16,589 of 17,186 rows in dataset/lines.jsonl are aligned=False
# (96.5%), and Pearson r(crop width, label length) is 0.194 where correctly
# paired line data would be >=0.8. Do not run this script.
#!/usr/bin/env python3
"""
Extract text-line crops from labeled images for TrOCR training.

Uses Surya text detection for line bboxes, aligns with label text lines.

Usage:
    python scripts/extract_lines.py
    python scripts/extract_lines.py --dataset dataset/dataset_clean.jsonl --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

LINES_DIR = PROJECT_ROOT / "dataset" / "lines"
LINES_JSONL = PROJECT_ROOT / "dataset" / "lines.jsonl"
MISMATCH_LOG = PROJECT_ROOT / "dataset" / "lines_mismatch.jsonl"


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def _bbox_xyxy(bbox: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    return x1, y1, x2, y2


def detect_line_bboxes(rgb: np.ndarray) -> list[list[float]]:
    from surya.detection import batch_text_detection
    from pipeline.layout import det_model, det_processor

    pil = Image.fromarray(rgb)
    result = batch_text_detection([pil], det_model, det_processor)[0]
    boxes = [box.bbox for box in result.bboxes]
    # Top-to-bottom, right-to-left for Persian
    boxes.sort(key=lambda b: (b[1], -b[0]))
    return boxes


def split_label_lines(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n")]
    return [ln for ln in lines if ln]


def extract_from_record(
    record: dict,
    *,
    lines_dir: Path,
    min_line_height: int = 8,
) -> tuple[list[dict], dict | None]:
    rel_path = record["image_path"]
    image_path = PROJECT_ROOT / rel_path
    if not image_path.exists():
        return [], {"image_path": rel_path, "error": "file_missing"}

    rgb = _load_rgb(image_path)
    img_h, img_w = rgb.shape[:2]
    label_lines = split_label_lines(str(record.get("text") or ""))
    if not label_lines:
        return [], {"image_path": rel_path, "error": "empty_label"}

    bboxes = detect_line_bboxes(rgb)
    if not bboxes:
        return [], {"image_path": rel_path, "error": "no_lines_detected", "label_lines": len(label_lines)}

    # Align: use min count; flag mismatch
    count = min(len(bboxes), len(label_lines))
    mismatch = None
    if len(bboxes) != len(label_lines):
        mismatch = {
            "image_path": rel_path,
            "detected_lines": len(bboxes),
            "label_lines": len(label_lines),
            "needs_review": True,
        }

    out_records: list[dict] = []
    lines_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        x1, y1, x2, y2 = _bbox_xyxy(bboxes[i], img_w, img_h)
        if y2 - y1 < min_line_height or x2 - x1 < 4:
            continue

        crop = rgb[y1:y2, x1:x2]
        line_id = uuid.uuid4().hex[:12]
        line_name = f"{line_id}.jpg"
        line_path = lines_dir / line_name
        Image.fromarray(crop).save(line_path, quality=92)

        out_records.append({
            "image_path": f"dataset/lines/{line_name}",
            "text": label_lines[i],
            "source_image": rel_path,
            "line_index": i,
            "bbox": [x1, y1, x2, y2],
            "aligned": len(bboxes) == len(label_lines),
        })

    return out_records, mismatch


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract line crops for TrOCR")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset" / "dataset_clean.jsonl")
    parser.add_argument("--lines-dir", type=Path, default=LINES_DIR)
    parser.add_argument("--output", type=Path, default=LINES_JSONL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true", help="Append to lines.jsonl")
    args = parser.parse_args()

    if not args.dataset.exists():
        # Fallback to dataset.jsonl
        fallback = PROJECT_ROOT / "dataset" / "dataset.jsonl"
        if fallback.exists():
            args.dataset = fallback
        else:
            print(f"Dataset not found: {args.dataset}")
            return 1

    records: list[dict] = []
    with args.dataset.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    all_lines: list[dict] = []
    mismatches: list[dict] = []

    for i, record in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {record.get('image_path')}")
        lines, mismatch = extract_from_record(record, lines_dir=args.lines_dir)
        all_lines.extend(lines)
        if mismatch:
            mismatches.append(mismatch)
        print(f"  → {len(lines)} lines")

    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for rec in all_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with MISMATCH_LOG.open("a" if args.append else "w", encoding="utf-8") as f:
        for rec in mismatches:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal lines: {len(all_lines)} | Mismatches: {len(mismatches)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
