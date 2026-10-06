"""
Layout / line detection using Surya with an OpenCV projection-profile fallback.

Known failure mode this module addresses: treating a full page as a single text
block, which destroys line/word segmentation for OCR. Default behaviour is now
**line-level** boxes from Surya text detection (plus table/figure regions from
the layout model). Set ``legacy_full_blocks=True`` only for before/after eval.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_detection_model
from surya.model.detection.model import load_processor as load_detection_processor
from surya.settings import settings

Block = dict[str, object]

# HuggingFace cache inside project directory (before model download).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))


def load_layout_model():
    return load_detection_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)


def load_layout_processor():
    return load_detection_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)


_SURYA_LABEL_TO_TYPE: dict[str, str] = {
    "Table": "table",
    "Picture": "figure",
    "Figure": "figure",
    "Text": "text",
    "List-item": "list",
    "List": "list",
    "Title": "text",
    "Section-header": "text",
    "Caption": "text",
    "Page-header": "text",
    "Page-footer": "text",
    "Footnote": "text",
    "Formula": "figure",
}

# Detection + layout load at import time. Reading-order is optional: newer
# ``transformers`` releases remove symbols Surya's ordering encoder still imports.
det_processor = load_detection_processor()
det_model = load_detection_model()
layout_processor = load_layout_processor()
layout_model = load_layout_model()

order_processor = None
order_model = None
try:
    from surya.model.ordering.model import load_model as load_order_model
    from surya.model.ordering.processor import load_processor as load_order_processor

    order_processor = load_order_processor()
    order_model = load_order_model()
except Exception as _order_exc:  # noqa: BLE001 — environment/version skew
    _debug_boot = os.environ.get("DEBUG_LAYOUT") == "1"
    if _debug_boot:
        print(f"[layout] ordering model unavailable ({_order_exc}); using RTL sort")


def _debug(msg: str) -> None:
    if os.environ.get("DEBUG_LAYOUT") == "1":
        print(f"[layout] {msg}")


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(image)


def _map_surya_label(label: str) -> str:
    return _SURYA_LABEL_TO_TYPE.get(label, "text")


def _bbox_xyxy(bbox: list[float], *, w: int, h: int) -> list[int]:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _sort_blocks_rtl(blocks: list[Block], *, band_px: int = 24) -> list[Block]:
    """Top-to-bottom by band, then right-to-left (Persian reading order)."""
    band = max(1, band_px)

    def key(b: Block) -> tuple:
        x1, y1, x2, y2 = b["bbox"]  # type: ignore[misc]
        return (int(y1) // band, -int(x2))

    return sorted(blocks, key=key)


def _area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _is_near_full_page(bbox: list[int], w: int, h: int, *, frac: float = 0.85) -> bool:
    return _area(bbox) >= frac * w * h


# Geometry gates shared with scripts/extract_and_align_lines.py — reject noise,
# page rules, and portrait fragments that are not text lines.
_MIN_LINE_HEIGHT = 24
_MAX_LINE_ASPECT = 16.0


def _is_plausible_text_line(bbox: list[int]) -> bool:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    if bh < _MIN_LINE_HEIGHT:
        return False
    if bh > bw:  # portrait → not a text line
        return False
    if bw / bh > _MAX_LINE_ASPECT:
        return False
    return True


def _surya_line_blocks(pil_image: Image.Image, image: np.ndarray) -> list[Block]:
    h, w = image.shape[:2]
    line_predictions = batch_text_detection([pil_image], det_model, det_processor)
    if not line_predictions or not line_predictions[0].bboxes:
        return []
    blocks: list[Block] = []
    for box in line_predictions[0].bboxes:
        bbox = _bbox_xyxy(box.bbox, w=w, h=h)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        conf = float(box.confidence if box.confidence is not None else 0.5)
        blocks.append({"type": "text", "bbox": bbox, "confidence": conf, "level": "line"})
    return _sort_blocks_rtl(blocks)


def _projection_profile_lines(
    image: np.ndarray,
    *,
    min_height: int = 16,
    min_width_frac: float = 0.15,
) -> list[Block]:
    """OpenCV horizontal projection-profile line segmentation fallback."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]
    # Dark ink on light paper → invert so ink is white for projection.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Suppress thin noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 80), 1))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    projection = bw.sum(axis=1).astype(np.float64)
    if projection.max() <= 0:
        return []
    thresh = 0.08 * float(projection.max())
    active = projection >= thresh

    bands: list[tuple[int, int]] = []
    start = None
    for y, on in enumerate(active):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None and h - start >= min_height:
        bands.append((start, h))

    min_width = int(min_width_frac * w)
    blocks: list[Block] = []
    for y1, y2 in bands:
        strip = bw[y1:y2, :]
        cols = strip.sum(axis=0)
        col_on = np.where(cols > 0)[0]
        if col_on.size == 0:
            continue
        x1, x2 = int(col_on[0]), int(col_on[-1]) + 1
        if x2 - x1 < min_width:
            continue
        pad = max(2, (y2 - y1) // 10)
        yy1, yy2 = max(0, y1 - pad), min(h, y2 + pad)
        blocks.append(
            {
                "type": "text",
                "bbox": [x1, yy1, x2, yy2],
                "confidence": 0.4,
                "level": "line",
                "source": "projection_profile",
            }
        )
    return _sort_blocks_rtl(blocks, band_px=min_height)


def _layout_regions(
    pil_image: Image.Image,
    image: np.ndarray,
    line_predictions,
    *,
    use_ordering: bool = True,
) -> list[Block]:
    """Non-line layout regions (tables/figures) + optional legacy full text blocks."""
    h, w = image.shape[:2]
    layout_predictions = batch_layout_detection(
        [pil_image],
        layout_model,
        layout_processor,
        line_predictions,
    )
    layout_pred = layout_predictions[0]
    if not layout_pred.bboxes:
        return []

    layout_bboxes = [box.bbox for box in layout_pred.bboxes]
    boxes_ordered = list(layout_pred.bboxes)
    if use_ordering and order_model is not None and order_processor is not None:
        try:
            from surya.ordering import batch_ordering

            order_predictions = batch_ordering(
                [pil_image],
                [layout_bboxes],
                order_model,
                order_processor,
            )
            order_pred = order_predictions[0]
            paired = list(zip(layout_pred.bboxes, order_pred.bboxes))
            paired.sort(key=lambda item: item[1].position)
            boxes_ordered = [layout_box for layout_box, _ in paired]
        except Exception as exc:
            _debug(f"reading order failed ({exc}); using RTL coordinate sort")
            boxes_ordered = list(layout_pred.bboxes)
    else:
        _debug("ordering skipped or unavailable; using RTL coordinate sort")

    blocks: list[Block] = []
    for box in boxes_ordered:
        mapped = _map_surya_label(box.label)
        bbox = _bbox_xyxy(box.bbox, w=w, h=h)
        conf = float(box.confidence if box.confidence is not None else 0.5)
        blocks.append(
            {
                "type": mapped,
                "bbox": bbox,
                "confidence": conf,
                "level": "region",
                "surya_label": box.label,
            }
        )
    return blocks


def _full_page_block(image: np.ndarray) -> list[Block]:
    h, w = image.shape[:2]
    return [
        {
            "type": "text",
            "bbox": [0, 0, w, h],
            "confidence": 0.5,
            "level": "page",
        }
    ]


def detect_blocks(
    image: np.ndarray,
    *,
    legacy_full_blocks: bool = False,
    use_projection_fallback: bool = True,
) -> list[Block]:
    """
    Detect layout/line blocks and return them in reading order.

    Returns list of dicts: type, bbox [x1, y1, x2, y2], confidence.

    Default (``legacy_full_blocks=False``): emit **line-level** text boxes from
    Surya detection, plus table/figure regions from the layout model. If line
    detection fails, fall back to an OpenCV projection profile, then to a
    single full-page block.
    """
    pil_image = _to_pil_rgb(image)
    h, w = image.shape[:2]

    line_predictions = batch_text_detection([pil_image], det_model, det_processor)
    # Line-level path sorts by coordinates; skip the slow reading-order model.
    region_blocks = _layout_regions(
        pil_image,
        image,
        line_predictions,
        use_ordering=legacy_full_blocks,
    )

    if legacy_full_blocks:
        if not region_blocks:
            _debug("legacy: no layout boxes; full-page fallback")
            return _full_page_block(image)
        # Historical behaviour: use layout regions as OCR blocks (often page-sized).
        out = [
            {
                "type": b["type"],
                "bbox": b["bbox"],
                "confidence": b["confidence"],
            }
            for b in region_blocks
        ]
        _debug(f"legacy total: {len(out)} blocks")
        return _sort_blocks_rtl(out)

    # Line-level path.
    line_blocks: list[Block] = []
    if line_predictions and line_predictions[0].bboxes:
        for box in line_predictions[0].bboxes:
            bbox = _bbox_xyxy(box.bbox, w=w, h=h)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            # Skip lines that are essentially the whole page (detector failure).
            if _is_near_full_page(bbox, w, h):
                continue
            if not _is_plausible_text_line(bbox):
                continue
            conf = float(box.confidence if box.confidence is not None else 0.5)
            line_blocks.append(
                {"type": "text", "bbox": bbox, "confidence": conf, "level": "line"}
            )
        line_blocks = _sort_blocks_rtl(line_blocks)

    if not line_blocks and use_projection_fallback:
        _debug("no usable Surya lines; trying projection-profile fallback")
        line_blocks = _projection_profile_lines(image)

    non_text = [b for b in region_blocks if b["type"] in ("table", "figure")]
    # Drop table/figure regions that duplicate a near-full-page box when we have lines.
    if line_blocks:
        non_text = [b for b in non_text if not _is_near_full_page(b["bbox"], w, h)]  # type: ignore[arg-type]

    blocks: list[Block] = []
    for b in line_blocks:
        blocks.append(
            {
                "type": "text",
                "bbox": b["bbox"],
                "confidence": b["confidence"],
                "level": b.get("level", "line"),
            }
        )
    for b in non_text:
        blocks.append(
            {
                "type": b["type"],
                "bbox": b["bbox"],
                "confidence": b["confidence"],
                "level": "region",
            }
        )

    if not blocks:
        _debug("no lines/regions; using full-page fallback block")
        return _full_page_block(image)

    blocks = _sort_blocks_rtl(blocks)
    _debug(f"final total: {len(blocks)} blocks (lines={len(line_blocks)}, regions={len(non_text)})")
    return blocks


def draw_debug_overlay(image: np.ndarray, blocks: list[Block]) -> np.ndarray:
    """Return a BGR image with detected boxes drawn (for debug dumps)."""
    if len(image.shape) == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        # Assume RGB input from pipeline; convert for OpenCV drawing.
        canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    colors = {
        "text": (40, 180, 40),
        "list": (40, 180, 40),
        "table": (40, 40, 220),
        "figure": (220, 140, 40),
    }
    for i, block in enumerate(blocks):
        x1, y1, x2, y2 = block["bbox"]  # type: ignore[misc]
        color = colors.get(str(block.get("type", "text")), (200, 200, 200))
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{i}:{block.get('type', '?')}"
        cv2.putText(
            canvas,
            label,
            (int(x1), max(12, int(y1) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_debug_overlays(
    image_paths: list[Path],
    out_dir: Path,
    *,
    legacy_full_blocks: bool = False,
    limit: int = 10,
) -> list[Path]:
    """Run detection on up to ``limit`` pages and save overlay PNGs."""
    from pipeline.preprocess import enhance_page_rgb

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in image_paths[:limit]:
        path = Path(path)
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        page = enhance_page_rgb(rgb)
        blocks = detect_blocks(page, legacy_full_blocks=legacy_full_blocks)
        overlay = draw_debug_overlay(page, blocks)
        tag = "legacy" if legacy_full_blocks else "lines"
        out_path = out_dir / f"{path.stem}_{tag}_n{len(blocks)}.png"
        cv2.imwrite(str(out_path), overlay)
        written.append(out_path)
        print(f"[layout-debug] {path.name}: {len(blocks)} blocks → {out_path}")
    return written
