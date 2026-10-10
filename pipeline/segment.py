"""Split a page into reading-order items: editable text lines and image-only blocks.

Line-level items are what make student edits usable as training pairs: every
edit maps to exactly one crop (see `.claude/skills/feedback-learning`).

Backends (env `SEGMENTER`):
- ``surya``  — Surya layout + text-line detection (best quality; needs surya-ocr,
  which pins transformers 4.45 and so cannot share an env with Qwen3-VL).
- ``opencv`` — morphology-based line finder; no ML deps, good enough for clean
  scans and the demo.
- ``auto``   — surya if importable, otherwise opencv (default).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Item = dict[str, Any]  # {"kind": "text"|"image", "block_type": str, "bbox": [x1, y1, x2, y2]}

# Vertical padding as a fraction of line height on each side; measured in
# REPORT_dataset_and_bakeoff.md §3 (0.075 recovered clipped ascenders/descenders
# without pulling in the neighbouring line).
LINE_PAD_FRACTION = 0.075


def _pad_line(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad = int(round((y2 - y1) * LINE_PAD_FRACTION))
    return [max(0, x1), max(0, y1 - pad), min(width, x2), min(height, y2 + pad)]


def _center_inside(line: list[int], block: list[int]) -> bool:
    cx = (line[0] + line[2]) / 2
    cy = (line[1] + line[3]) / 2
    return block[0] <= cx <= block[2] and block[1] <= cy <= block[3]


def order_lines_rtl(boxes: list[list[int]]) -> list[list[int]]:
    """Group boxes into rows by vertical overlap, rows top→bottom, boxes right→left."""
    if not boxes:
        return []
    rows: list[list[list[int]]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (box[1] + box[3]) / 2
        for row in rows:
            ref = row[0]
            if ref[1] <= cy <= ref[3]:
                row.append(box)
                break
        else:
            rows.append([box])
    ordered: list[list[int]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: -b[2]))
    return ordered


# ── OpenCV backend ──────────────────────────────────────────────────────────
def _opencv_line_boxes(image: np.ndarray) -> list[list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    # Answer sheets are often ruled; remove long horizontal/vertical rules so they
    # don't fuse every line on the page into one component.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 12), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 12)))
    rules = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel),
    )
    ink = cv2.subtract(binary, rules)

    # Join letters and words of one line horizontally, but not adjacent lines.
    join = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, width // 35), 3))
    joined = cv2.dilate(ink, join, iterations=1)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    min_h = max(8, height // 150)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        points = cv2.findNonZero(ink[y : y + h, x : x + w])
        if points is None:
            continue
        # Tight ink box: the dilated contour is wider than the text it covers.
        tx, ty, tw, th = cv2.boundingRect(points)
        x, y, w, h = x + tx, y + ty, tw, th
        if h < min_h or w < max(6, width // 200):
            continue
        if w * 4 < h:  # thin vertical sliver: leftover of a box border, not text
            continue
        if len(points) < 0.02 * w * h:
            continue
        boxes.append([x, y, x + w, y + h])
    return boxes


def _segment_opencv(image: np.ndarray) -> list[Item]:
    height, width = image.shape[:2]
    items: list[Item] = []
    boxes = _opencv_line_boxes(image)
    if boxes:
        median_h = float(np.median([b[3] - b[1] for b in boxes]))
    for box in order_lines_rtl(boxes):
        box_h = box[3] - box[1]
        # Very tall components are drawings/tables, not text lines.
        if box_h > 3 * median_h and (box[2] - box[0]) * box_h > 0.01 * width * height:
            items.append({"kind": "image", "block_type": "figure", "bbox": box})
        else:
            items.append({"kind": "text", "block_type": "text", "bbox": _pad_line(box, width, height)})
    return items


# ── Surya backend ───────────────────────────────────────────────────────────
def _segment_surya(image: np.ndarray) -> list[Item]:
    from pipeline.crop import should_crop
    from pipeline.layout import detect_blocks, detect_text_lines

    height, width = image.shape[:2]
    blocks = detect_blocks(image)
    lines = detect_text_lines(image)
    used: set[int] = set()
    items: list[Item] = []

    for block in blocks:
        bbox = list(block["bbox"])
        if should_crop(block):
            items.append({"kind": "image", "block_type": block.get("type"), "bbox": bbox})
            used.update(i for i, ln in enumerate(lines) if _center_inside(ln, bbox))
            continue
        inside = [i for i, ln in enumerate(lines) if i not in used and _center_inside(ln, bbox)]
        used.update(inside)
        for ln in order_lines_rtl([lines[i] for i in inside]):
            items.append(
                {"kind": "text", "block_type": block.get("type"), "bbox": _pad_line(ln, width, height)}
            )

    leftovers = [ln for i, ln in enumerate(lines) if i not in used]
    for ln in order_lines_rtl(leftovers):
        items.append({"kind": "text", "block_type": "text", "bbox": _pad_line(ln, width, height)})
    return items


def segmenter_name() -> str:
    choice = os.environ.get("SEGMENTER", "auto").lower()
    if choice != "auto":
        return choice
    try:
        import surya  # noqa: F401
    except Exception:
        return "opencv"
    return "surya"


def segment_page(image_rgb: np.ndarray) -> list[Item]:
    """Return reading-order items for one RGB page."""
    name = segmenter_name()
    if name == "surya":
        try:
            items = _segment_surya(image_rgb)
            if items:
                return items
        except Exception as exc:  # model load / inference failure
            logger.warning("Surya segmentation failed (%s); falling back to OpenCV", exc)
    return _segment_opencv(image_rgb)
