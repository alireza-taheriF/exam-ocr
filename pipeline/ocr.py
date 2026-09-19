"""
Text extraction for exam blocks using a Vision-Language Model (VLM) API.

Layout detection still uses Surya; text/list blocks are sent to an
OpenAI-compatible vision API (e.g. DeepSeek).
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import RateLimitError
from pipeline.openai_compat import (
    create_openai_client,
    resolve_openai_api_key,
    resolve_openai_base_url,
)
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline.crop import crop_block, should_crop

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

_OCR_BLOCK_TYPES = frozenset({"text", "list"})
_OCR_MIN_TEXT_LENGTH = 2
_OCR_MIN_CROP_HEIGHT = int(os.environ.get("OCR_MIN_CROP_HEIGHT", "64"))
_OCR_UPSCALE_FACTOR = float(os.environ.get("OCR_UPSCALE_FACTOR", "2.0"))
_OCR_REQUEST_DELAY_SEC = float(os.environ.get("OCR_REQUEST_DELAY_SEC", "2.5"))
_PLACEHOLDER_API_KEYS = frozenset({"", "your_api_key_here", "changeme", "replace_me"})

_SYSTEM_PROMPT = (
    "تو یک موتور OCR بسیار دقیق برای زبان فارسی هستی. "
    "تصویر بخشی از یک برگه امتحان دانشگاهی است و ممکن است چاپی یا دست‌نویس باشد. "
    "متن را دقیقاً همان‌طور که در تصویر نوشته شده استخراج کن؛ "
    "از اصلاح املا، تکمیل جمله، ترجمه یا توضیح اضافه خودداری کن. "
    "اعداد، علائم نگارشی، پرانتز و نقطه‌گذاری را حفظ کن. "
    "اگر بخشی کاملاً ناخوانا است فقط همان بخش را با [ناخوانا] علامت بزن. "
    "فقط متن استخراج‌شده را برگردان، بدون مقدمه یا جمع‌بندی."
)

_USER_PROMPT = (
    "متن فارسی داخل این تصویر (چاپی یا دست‌نویس) را خط‌به‌خط و دقیق استخراج کن."
)

_VLM_CLIENT: object | None = None
_VLM_CONFIG_ERROR: str | None = None


def _get_api_key() -> str:
    return resolve_openai_api_key(dict(os.environ))


def _validate_vlm_config() -> str | None:
    """Return an error message when the VLM API is not configured."""
    api_key = _get_api_key()
    if not api_key or api_key.lower() in _PLACEHOLDER_API_KEYS:
        return (
            "VLM_API_KEY is missing or still set to the placeholder value. "
            "Copy .env.example to .env and set a valid API key."
        )
    return None


def _get_vlm_client():
    global _VLM_CLIENT, _VLM_CONFIG_ERROR

    config_error = _validate_vlm_config()
    if config_error:
        _VLM_CONFIG_ERROR = config_error
        raise ValueError(config_error)

    if _VLM_CLIENT is not None:
        return _VLM_CLIENT

    api_base = resolve_openai_base_url(dict(os.environ))
    _VLM_CLIENT = create_openai_client(api_key=_get_api_key(), base_url=api_base)
    _VLM_CONFIG_ERROR = None
    return _VLM_CLIENT


def _to_pil_rgb(image: np.ndarray) -> Image.Image:
    """Convert grayscale, BGR, or RGB numpy image to PIL RGB."""
    if len(image.shape) == 2:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    return Image.fromarray(image)


def _crop_region(pil_image: Image.Image, bbox: list) -> Image.Image:
    x1, y1, x2, y2 = bbox
    img_w, img_h = pil_image.size
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop bbox: {bbox}")
    return pil_image.crop((x1, y1, x2, y2))


def _pil_to_base64(image: Image.Image, *, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _clean_vlm_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w*)\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _message_content_to_text(content: object) -> str:
    """Normalize OpenAI-compatible message content (str or multimodal parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _clean_vlm_output(content)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text = part.get("text")
                    if text:
                        parts.append(str(text))
            elif isinstance(part, str):
                parts.append(part)
        return _clean_vlm_output("\n".join(parts))
    return _clean_vlm_output(str(content))


def _is_illegible_text(text: str) -> bool:
    stripped = text.strip()
    return not stripped or len(stripped) < _OCR_MIN_TEXT_LENGTH


def _mark_illegible(block: dict, original_image: np.ndarray, *, reason: str | None = None) -> None:
    block["illegible"] = True
    block["text"] = None
    block["crop_image"] = crop_block(original_image, block["bbox"], rgb_input=True)
    if reason:
        block["vlm_error"] = reason


def _prepare_ocr_region(region: Image.Image) -> Image.Image:
    """Upscale small handwriting regions so the VLM can read fine strokes."""
    width, height = region.size
    if height >= _OCR_MIN_CROP_HEIGHT and width >= _OCR_MIN_CROP_HEIGHT:
        return region

    scale = max(
        _OCR_UPSCALE_FACTOR,
        _OCR_MIN_CROP_HEIGHT / max(height, 1),
        _OCR_MIN_CROP_HEIGHT / max(width, 1),
    )
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return region.resize(new_size, Image.Resampling.LANCZOS)


@retry(
    retry=retry_if_exception_type(RateLimitError),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=60),
)
def extract_text_with_vlm(base64_image: str) -> str:
    """
    Send a base64-encoded image to the configured VLM API and return extracted text.

    Reads VLM_API_BASE, VLM_API_KEY, and optional VLM_MODEL from the environment.
    """
    client = _get_vlm_client()
    model = os.environ.get("VLM_MODEL", "deepseek-chat")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ],
    )

    if not response.choices:
        return ""
    message = response.choices[0].message
    if message is None:
        return ""
    return _message_content_to_text(message.content)


def get_ocr_status() -> dict[str, str | bool]:
    """Return OCR engine configuration for health checks (no secrets)."""
    config_error = _validate_vlm_config()
    return {
        "engine": "vlm",
        "configured": config_error is None,
        "model": os.environ.get("VLM_MODEL", "deepseek-chat"),
        "api_base": os.environ.get("VLM_API_BASE", "https://api.deepseek.com"),
        "message": config_error or "ready",
    }


def _wants_ocr(block: dict) -> bool:
    """True when this block should be sent to the VLM (not embedded as image only)."""
    if should_crop(block):
        return False
    block_type = block.get("type")
    if block_type in _OCR_BLOCK_TYPES:
        return True
    # Surya often labels small handwriting crops as Figure/Picture.
    return block_type == "figure"


def run_ocr(original_image: np.ndarray, blocks: list[dict]) -> list[dict]:
    """
    For each block:
    - table/figure: crop directly (no VLM call)
    - text/list: crop region → base64 → VLM API; fallback to crop on failure
    """
    pil_image = _to_pil_rgb(original_image)
    config_error = _validate_vlm_config()
    if config_error:
        logger.warning("%s Text blocks will be exported as crops.", config_error)

    ocr_calls = 0
    for block in blocks:
        if should_crop(block):
            block["text"] = None
            block["illegible"] = False
            block["crop_image"] = crop_block(original_image, block["bbox"], rgb_input=True)
            continue

        block_type = block.get("type")
        if not _wants_ocr(block):
            block["text"] = None
            block["crop_image"] = None
            block["illegible"] = False
            continue

        if config_error:
            _mark_illegible(block, original_image, reason=config_error)
            continue

        try:
            if ocr_calls > 0 and _OCR_REQUEST_DELAY_SEC > 0:
                time.sleep(_OCR_REQUEST_DELAY_SEC)
            region = _crop_region(pil_image, block["bbox"])
            region = _prepare_ocr_region(region)
            base64_image = _pil_to_base64(region)
            text = extract_text_with_vlm(base64_image)
            ocr_calls += 1

            if _is_illegible_text(text):
                logger.warning("VLM returned empty or short text for block %s", block["bbox"])
                _mark_illegible(block, original_image, reason="empty_or_short_vlm_output")
                continue

            block["text"] = text
            block["crop_image"] = None
            block["illegible"] = False

        except Exception as exc:
            logger.warning("VLM OCR failed on block %s: %s", block["bbox"], exc)
            _mark_illegible(block, original_image, reason=str(exc))

    return blocks
