"""
Smart cropping for exam blocks that should be embedded instead of OCR'd.

Tables and figures are cropped from the original color image and lightly
enhanced for Word/PDF export. Low-quality OCR regions are cropped in ocr.py.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Block = dict[str, object]

# Figures larger than this area are embedded; smaller ones are often mislabeled handwriting.
_LARGE_FIGURE_AREA = 50_000
_LARGE_FIGURE_SIDE = 400


def should_crop(block: Block) -> bool:
    """Return True if this block should be cropped instead of OCR'd."""
    block_type = block.get("type")
    if block_type == "table":
        return True
    if block_type != "figure":
        return False

    bbox = block.get("bbox") or [0, 0, 0, 0]
    width = max(0, int(bbox[2]) - int(bbox[0]))
    height = max(0, int(bbox[3]) - int(bbox[1]))
    area = width * height
    if area < _LARGE_FIGURE_AREA and max(width, height) < _LARGE_FIGURE_SIDE:
        return False
    return True


def _clamp_bbox(bbox: list[int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    """Apply padding and clamp crop coordinates to image bounds (xyxy format)."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(width, int(x2) + padding)
    y2 = min(height, int(y2) + padding)
    return x1, y1, x2, y2


def _auto_brightness_contrast(pil_image: Image.Image) -> Image.Image:
    """
    Auto-adjust brightness and contrast using robust percentile stretching.

    Helps mobile photos and uneven scans look cleaner when embedded in exports.
    """
    arr = np.asarray(pil_image).astype(np.float32)
    if arr.size == 0:
        return pil_image

    low = np.percentile(arr, 2)
    high = np.percentile(arr, 98)
    if high - low < 1.0:
        return pil_image

    stretched = np.clip((arr - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    adjusted = Image.fromarray(stretched)

    adjusted = ImageEnhance.Brightness(adjusted).enhance(1.03)
    adjusted = ImageEnhance.Contrast(adjusted).enhance(1.08)
    return adjusted


def _sharpen(pil_image: Image.Image) -> Image.Image:
    """Apply a light sharpen pass without amplifying compression noise too much."""
    return ImageEnhance.Sharpness(pil_image).enhance(1.25).filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=90, threshold=3)
    )


def crop_block(
    original_image: np.ndarray,
    bbox: list,
    padding: int = 10,
    *,
    rgb_input: bool = False,
) -> Image.Image:
    """
    Crop and enhance a layout block from the original (non-thresholded) image.

    Args:
        original_image: Source page in BGR, RGB, or grayscale.
        bbox: Bounding box as [x1, y1, x2, y2].
        padding: Extra pixels around the bbox, clamped to image bounds.

    Returns:
        PIL Image (RGB) ready for embedding in Word/PDF output.
    """
    if len(original_image.shape) < 2:
        raise ValueError("original_image must be a 2D or 3D numpy array")

    height, width = original_image.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(list(bbox), width, height, padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after padding/clamping: {bbox}")

    cropped = original_image[y1:y2, x1:x2]
    if len(cropped.shape) == 2:
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB))
    elif rgb_input:
        pil_image = Image.fromarray(cropped)
    else:
        pil_image = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_image = _auto_brightness_contrast(pil_image)
    pil_image = _sharpen(pil_image)
    return pil_image
