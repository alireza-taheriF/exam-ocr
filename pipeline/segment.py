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
from pathlib import Path
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
def _paper_mask(gray: np.ndarray) -> np.ndarray:
    """255 inside the sheet of paper, 0 on the table / scanner bed around it.

    Phone photos of notebooks show the desk and the spiral binding; without this
    every ring and the page edge become "text lines" (dataset/reports/segmentation).
    """
    height, width = gray.shape[:2]
    small = cv2.resize(gray, (max(1, width // 8), max(1, height // 8)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (9, 9), 0)
    _, bright = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n <= 1:
        return np.full_like(gray, 255)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 0.5 * small.size:
        return np.full_like(gray, 255)  # no clear sheet (e.g. dark scan): keep everything
    sheet = np.where(labels == biggest, 255, 0).astype(np.uint8)
    # Fill holes (dark text blocks inside the sheet), then pull the edge in so the
    # rings and the shadowed page border stay outside.
    contours, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = cv2.drawContours(np.zeros_like(sheet), contours, -1, 255, thickness=cv2.FILLED)
    margin = max(3, sheet.shape[1] // 40)
    sheet = cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_RECT, (margin, margin)))
    return cv2.resize(sheet, (width, height), interpolation=cv2.INTER_NEAREST)


def _remove_rules(binary: np.ndarray) -> np.ndarray:
    """Remove ruled-paper lines and margin lines, which fuse text lines together.

    Ruling on photographed notebooks is faint and broken by the threshold, so
    short gaps are closed before keeping only long straight runs.
    """
    height, width = binary.shape[:2]
    h_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 120), 1)))
    h_rules = cv2.morphologyEx(h_closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 8), 1)))
    v_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, height // 120))))
    v_rules = cv2.morphologyEx(v_closed, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 8))))
    rules = cv2.bitwise_or(h_rules, v_rules)
    # Rules are a few pixels thick and anti-aliased; widen the mask slightly.
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return cv2.subtract(binary, rules)


def _split_merged(ink: np.ndarray, box: list[int], line_h: float) -> list[list[int]]:
    """Split a box that spans several text lines at the valleys of its row profile."""
    x1, y1, x2, y2 = box
    if y2 - y1 < 1.8 * line_h:
        return [box]
    profile = (ink[y1:y2, x1:x2] > 0).sum(axis=1).astype(float)
    k = max(3, int(line_h // 4) | 1)
    profile = np.convolve(profile, np.ones(k) / k, mode="same")
    on = profile > 0.12 * profile.max()
    parts: list[list[int]] = []
    start = None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 0.35 * line_h:
                parts.append([x1, y1 + start, x2, y1 + i])
            start = None
    if len(parts) < 2:
        return [box]
    out = []
    for px1, py1, px2, py2 in parts:  # re-tighten each part horizontally
        cols = np.nonzero((ink[py1:py2, px1:px2] > 0).any(axis=0))[0]
        if cols.size:
            out.append([px1 + int(cols[0]), py1, px1 + int(cols[-1]) + 1, py2])
    return out or [box]


def _merge_same_line(boxes: list[list[int]], line_h: float) -> list[list[int]]:
    """Merge pieces of one written line that the horizontal join left apart.

    Two boxes belong to one line when they share most of their height and the
    horizontal gap between them is a few line heights at most (a word, a dash or
    a bracket tip separated from the rest). Without this a line becomes two or
    three overlapping crops, each only part of the text.
    """
    boxes = [list(b) for b in boxes]
    max_gap = 2.5 * line_h

    def same_line(a: list[int], b: list[int]) -> bool:
        shared = min(a[3], b[3]) - max(a[1], b[1])
        if shared < 0.5 * min(a[3] - a[1], b[3] - b[1]):
            return False
        gap = max(a[0], b[0]) - min(a[2], b[2])  # < 0 when they overlap horizontally
        return gap <= max_gap

    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if same_line(boxes[i], boxes[j]):
                    a, b = boxes[i], boxes.pop(j)
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    merged = True
                    break
            if merged:
                break
    return boxes


def _ink_mask(image: np.ndarray) -> np.ndarray:
    """Binary ink (text and drawings) inside the sheet, with ruling removed."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    binary = cv2.bitwise_and(binary, _paper_mask(gray))
    return _remove_rules(binary)


def _opencv_line_boxes(image: np.ndarray) -> list[list[int]]:
    height, width = image.shape[:2]
    ink = _ink_mask(image)

    # Drop paper-texture speckle: components far smaller than a pen dot.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(6, (width // 600) ** 2)
    small = np.nonzero(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    if small.size:
        ink[np.isin(labels, small)] = 0
    # Line-height prior from glyph-cluster heights; independent of how well the
    # line boxes below come out (merged blocks would otherwise inflate it).
    glyph_h = stats[1:, cv2.CC_STAT_HEIGHT][
        (stats[1:, cv2.CC_STAT_AREA] >= 4 * min_area) & (stats[1:, cv2.CC_STAT_HEIGHT] < 0.05 * height)]
    glyph_line_h = 1.5 * float(np.percentile(glyph_h, 90)) if glyph_h.size else float("inf")

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

    boxes = _drop_binding_holes(boxes, width)
    if boxes:
        # Typical line height = median of boxes that look like real lines (wide
        # and wider than tall; short dashes and stray strokes would bias it low).
        # Merged blocks are then split at their row-profile valleys.
        flat = [b[3] - b[1] for b in boxes
                if (b[2] - b[0]) > max(2 * (b[3] - b[1]), 0.08 * width)]
        line_h = float(np.median(flat)) if flat else float(np.median([b[3] - b[1] for b in boxes]))
        line_h = min(line_h, glyph_line_h)
        boxes = [part for b in boxes for part in _split_merged(ink, b, line_h)]
        boxes = _absorb_fragments(boxes, line_h)
        boxes = _merge_same_line(boxes, line_h)
    return boxes


def _drop_binding_holes(boxes: list[list[int]], width: int) -> list[list[int]]:
    """Remove spiral-binding rings and punch holes: small, compact blobs in the
    outer side bands that repeat down the page in one column (≥4 of them)."""
    band, col_tol = 0.12 * width, 0.015 * width

    def candidate(b: list[int]) -> bool:
        w, h = b[2] - b[0], b[3] - b[1]
        return (b[2] < band or b[0] > width - band) and w < 0.04 * width and 0.4 * w <= h <= 4 * w

    cands = [b for b in boxes if candidate(b)]
    holes = set()
    for b in cands:
        cx = (b[0] + b[2]) / 2
        column = [c for c in cands if abs((c[0] + c[2]) / 2 - cx) <= col_tol]
        if len(column) >= 4:
            holes.add(id(b))
    return [b for b in boxes if id(b) not in holes]


def _absorb_fragments(boxes: list[list[int]], line_h: float) -> list[list[int]]:
    """Merge stray strokes (an ascender cut off by rule removal, a dot) into the
    line they touch; drop the ones that touch nothing and are too small to be a word."""
    def is_fragment(b: list[int]) -> bool:
        return (b[3] - b[1]) < line_h and (b[2] - b[0]) < 1.5 * line_h

    def inside(a: list[int], b: list[int]) -> float:
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = (a[2] - a[0]) * (a[3] - a[1])
        return ix * iy / area if area else 1.0

    # Boxes mostly inside a larger box (a word whose dilation did not touch the
    # rest of its line) are folded into it, largest first.
    lines: list[list[int]] = []
    for b in sorted((list(b) for b in boxes if not is_fragment(b)),
                    key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        host = next((h for h in lines if inside(b, h) >= 0.6), None)
        if host is None:
            lines.append(b)
        else:
            host[:] = [min(host[0], b[0]), min(host[1], b[1]), max(host[2], b[2]), max(host[3], b[3])]
    for f in (b for b in boxes if is_fragment(b)):
        best, best_gap = None, 0.6 * line_h
        for b in lines:
            if min(f[2], b[2]) - max(f[0], b[0]) <= 0:  # no horizontal overlap
                continue
            gap = max(0, max(f[1], b[1]) - min(f[3], b[3]))
            if gap <= best_gap:
                best, best_gap = b, gap
        if best is not None:
            best[0], best[1] = min(best[0], f[0]), min(best[1], f[1])
            best[2], best[3] = max(best[2], f[2]), max(best[3], f[3])
        elif (f[2] - f[0]) >= line_h:  # short word on its own (e.g. a heading)
            lines.append(list(f))
    return lines


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


# ── Surya via subprocess ────────────────────────────────────────────────────
SURYA_PYTHON = Path(os.environ.get("SURYA_PYTHON", Path(__file__).resolve().parent.parent / ".venv-surya" / "bin" / "python"))
SURYA_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "surya_lines.py"


def _surya_cli_boxes(image: np.ndarray) -> list[list[int]]:
    """Surya text-line boxes from the isolated environment (Surya 0.6 needs
    transformers 4.45; the Qwen engine needs 5). The script prints the JSON on
    its last stdout line, after the library's own log lines."""
    import json
    import subprocess
    import tempfile

    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "page.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        proc = subprocess.run([str(SURYA_PYTHON), str(SURYA_SCRIPT), str(path)],
                              capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"surya_lines failed: {proc.stderr[-300:]}")
    return json.loads([l for l in proc.stdout.splitlines() if l.strip()][-1])


def _has_ink(ink: np.ndarray, box: list[int], min_fraction: float = 0.004) -> bool:
    x1, y1, x2, y2 = box
    region = ink[max(0, y1):y2, max(0, x1):x2]
    return region.size > 0 and float((region > 0).mean()) >= min_fraction


def _covered(box: list[int], others: list[list[int]], fraction: float = 0.3) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    for o in others:
        ix = max(0, min(box[2], o[2]) - max(box[0], o[0]))
        iy = max(0, min(box[3], o[3]) - max(box[1], o[1]))
        if area and ix * iy / area >= fraction:
            return True
    return False


def _segment_surya_cli(image: np.ndarray) -> list[Item]:
    """Surya line boxes, completed by OpenCV for lines Surya missed.

    Surya finds whole lines on photographed notebooks where the OpenCV
    heuristics fragment them, but it also skips some lines and returns empty
    strips along the ruling. So: drop Surya boxes without ink, then add OpenCV
    boxes that Surya does not cover.
    """
    height, width = image.shape[:2]
    ink = _ink_mask(image)
    surya = [b for b in _surya_cli_boxes(image)
             if b[2] - b[0] > 6 and b[3] - b[1] > 6 and _has_ink(ink, b)]
    if surya:
        surya = _merge_same_line(surya, float(np.median([b[3] - b[1] for b in surya])))
    # Only whole lines Surya did not touch: no overlap with its boxes at all, and
    # wide enough to be a line (small leftovers next to a Surya line are fragments).
    line_h = float(np.median([b[3] - b[1] for b in surya])) if surya else 0.0
    extra = [b for b in _opencv_line_boxes(image)
             if not _covered(b, surya, fraction=0.02) and (b[2] - b[0]) >= 3 * line_h]
    return [{"kind": "text", "block_type": "text", "bbox": _pad_line(b, width, height)}
            for b in order_lines_rtl(surya + extra)]


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
    if name == "surya_cli":
        try:
            items = _segment_surya_cli(image_rgb)
            if items:
                return items
        except Exception as exc:
            logger.warning("Surya subprocess segmentation failed (%s); falling back to OpenCV", exc)
    return _segment_opencv(image_rgb)
