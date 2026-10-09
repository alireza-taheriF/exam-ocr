"""Resize line crops before VLM inference (avoids token truncation on wide images)."""

from __future__ import annotations

from PIL import Image


def resize_for_vlm(img: Image.Image, max_long_side: int = 1024) -> Image.Image:
    """Downscale so max(width, height) <= max_long_side (aspect preserved)."""
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)
