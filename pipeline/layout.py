"""
Layout block detection using Surya deep-learning models.

Uses text detection + layout segmentation + reading-order models instead of
OpenCV heuristics.
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
from surya.model.ordering.model import load_model as load_order_model
from surya.model.ordering.processor import load_processor as load_order_processor
from surya.ordering import batch_ordering
from surya.settings import settings

Block = dict[str, object]

# HuggingFace cache inside project directory (before model download).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HF_CACHE = _PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

# Surya layout/detection share the same loader; expose names requested by the pipeline.
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

# Load models once at import time.
det_processor = load_detection_processor()
det_model = load_detection_model()
layout_processor = load_layout_processor()
layout_model = load_layout_model()
order_processor = load_order_processor()
order_model = load_order_model()


def _debug(msg: str) -> None:
    if os.environ.get("DEBUG_LAYOUT") == "1":
        print(f"[layout] {msg}")


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    if len(image.shape) == 2:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(image)


def _map_surya_label(label: str) -> str:
    return _SURYA_LABEL_TO_TYPE.get(label, "text")


def _bbox_xyxy(bbox: list[float]) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [int(x1), int(y1), int(x2), int(y2)]


def _sort_blocks_rtl(blocks: list[Block]) -> list[Block]:
    """Fallback sort: top-to-bottom, right-to-left."""
    return sorted(blocks, key=lambda b: (b["bbox"][1], -b["bbox"][0]))


def detect_blocks(image: np.ndarray) -> list[Block]:
    """
    Detect layout blocks with Surya and return them in reading order.

    Returns list of dicts: type, bbox [x1, y1, x2, y2], confidence.
    """
    pil_image = _to_pil_rgb(image)

    line_predictions = batch_text_detection([pil_image], det_model, det_processor)
    layout_predictions = batch_layout_detection(
        [pil_image],
        layout_model,
        layout_processor,
        line_predictions,
    )
    layout_pred = layout_predictions[0]

    if not layout_pred.bboxes:
        _debug("no layout boxes detected; using full-page fallback block")
        height, width = image.shape[:2]
        return [
            {
                "type": "text",
                "bbox": [0, 0, width, height],
                "confidence": 0.5,
            }
        ]

    layout_bboxes = [box.bbox for box in layout_pred.bboxes]
    try:
        order_predictions = batch_ordering(
            [pil_image],
            [layout_bboxes],
            order_model,
            order_processor,
        )
        order_pred = order_predictions[0]
        paired = list(zip(layout_pred.bboxes, order_pred.bboxes))
        paired.sort(key=lambda item: item[1].position)
    except Exception as exc:
        _debug(f"reading order failed ({exc}); using RTL coordinate sort")
        paired = [(box, None) for box in layout_pred.bboxes]
        blocks = [
            {
                "type": _map_surya_label(box.label),
                "bbox": _bbox_xyxy(box.bbox),
                "confidence": float(box.confidence if box.confidence is not None else 0.5),
            }
            for box in layout_pred.bboxes
        ]
        _debug(f"final total: {len(blocks)} blocks")
        return _sort_blocks_rtl(blocks)

    blocks: list[Block] = []
    for layout_box, order_box in paired:
        confidence = float(layout_box.confidence if layout_box.confidence is not None else 0.5)
        blocks.append(
            {
                "type": _map_surya_label(layout_box.label),
                "bbox": _bbox_xyxy(order_box.bbox),
                "confidence": confidence,
            }
        )

    _debug(f"final total: {len(blocks)} blocks")
    return blocks


def detect_text_lines(image: np.ndarray) -> list[list[int]]:
    """Return Surya text-line boxes [x1, y1, x2, y2] (unordered)."""
    pil_image = _to_pil_rgb(image)
    line_predictions = batch_text_detection([pil_image], det_model, det_processor)
    return [_bbox_xyxy(box.bbox) for box in line_predictions[0].bboxes]
