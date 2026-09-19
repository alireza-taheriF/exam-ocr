#!/usr/bin/env python3
"""
Auto-label Persian handwritten/printed images using OpenAI vision API.

Reads images from dataset/raw_images (and optionally ingests PDFs from
dataset/raw_pdfs/) and appends {"image_path": "...", "text": "..."} to
dataset/dataset.jsonl.

Setup (OpenAI only):
    OPENAI_API_KEY=sk-proj-...  in .env
    python scripts/configure_openai_labeling.py
    python scripts/ingest_pdfs.py          # optional: PDF → page images
    python auto_labeler.py --resume --sleep 2
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pipeline.openai_compat import (
    AGENTROUTER_SYSTEM_PROMPT,
    AGENTROUTER_USER_PROMPT,
    create_openai_client,
    is_agentrouter,
    is_insufficient_quota,
)
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "dataset" / "raw_images"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "dataset" / "dataset.jsonl"
ERROR_LOG_FILE = PROJECT_ROOT / "dataset" / "label_errors.jsonl"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PLACEHOLDER_KEYS = frozenset({"", "your_api_key_here", "changeme", "replace_me"})
DEEPSEEK_HOSTS = ("api.deepseek.com", "deepseek.com")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENAI_BASE = "https://api.openai.com/v1"
MIN_LABEL_CHARS = 2

SYSTEM_PROMPT = (
    "تو یک متخصص استخراج متن (OCR/HTR) برای زبان فارسی هستی. "
    "متن دست‌نویس یا چاپی داخل این تصویر را دقیقاً همان‌طور که هست بخوان و تایپ کن. "
    "فقط و فقط متن استخراج‌شده را در خروجی برگردان و از نوشتن هرگونه توضیح اضافه، "
    "مقدمه یا نتیجه‌گیری خودداری کن."
)

USER_PROMPT = "متن داخل این تصویر را استخراج کن."

logger = logging.getLogger("auto_labeler")


@dataclass(frozen=True)
class VisionConfig:
    provider: str  # "openai" only when LABEL_OPENAI_ONLY=true
    api_key: str
    model: str
    base_url: str | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_placeholder(value: str) -> bool:
    return not value or value.lower() in PLACEHOLDER_KEYS


def _resolve_openai_base_url() -> str:
    from pipeline.openai_compat import resolve_openai_base_url

    return resolve_openai_base_url(dict(os.environ)) or OPENAI_BASE


def _openai_only_config() -> VisionConfig:
    """OpenAI-compatible API only — ignores OpenRouter, Gemini, FreeLLMAPI."""
    from pipeline.openai_compat import resolve_openai_api_key

    api_key = resolve_openai_api_key(dict(os.environ))
    if _is_placeholder(api_key):
        raise SystemExit(
            "JUSTWOKER_API_KEY or OPENAI_API_KEY is required.\n\n"
            "Add to .env:\n"
            "  JUSTWOKER_API_KEY=sk-...\n"
            "  JUSTWOKER_BASE_URL=https://api.justwoker.icu/v1\n\n"
            "Then run: python scripts/configure_openai_labeling.py"
        )
    model = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("VISION_MODEL", "").strip()
        or os.environ.get("VLM_MODEL", "").strip()
        or "gpt-4o-mini"
    )
    return VisionConfig(
        provider="openai",
        api_key=api_key,
        model=model,
        base_url=_resolve_openai_base_url(),
    )


def _load_config(provider_arg: str | None) -> VisionConfig:
    load_dotenv(PROJECT_ROOT / ".env")

    if _env_flag("LABEL_OPENAI_ONLY"):
        return _openai_only_config()

    # Legacy multi-provider path (disabled when LABEL_OPENAI_ONLY=true)
    if provider_arg == "gemini" or os.environ.get("LABEL_PROVIDER") == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()
        if _is_placeholder(gemini_key):
            raise SystemExit("GEMINI_API_KEY is missing.")
        return VisionConfig(provider="gemini", api_key=gemini_key, model=gemini_model)

    return _openai_only_config()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def _list_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _relative_image_path(image_path: Path) -> str:
    try:
        return image_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return image_path.as_posix()


def _load_completed_paths(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    completed: set[str] = set()
    with output_file.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at %s:%d", output_file, line_no)
                continue
            image_path = record.get("image_path")
            text = str(record.get("text") or "").strip()
            if image_path and len(text) >= MIN_LABEL_CHARS:
                completed.add(image_path)
    return completed


def _encode_image_base64(image_path: Path, *, max_dimension: int) -> tuple[str, str]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        longest = max(width, height)
        if longest > max_dimension:
            scale = max_dimension / longest
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=92)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded, "image/jpeg"


def _clean_model_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w*)\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _is_auth_error(exc: BaseException) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code == 401:
        return True
    detail = str(exc).lower()
    return "invalid_api_key" in detail or "incorrect api key" in detail


def _is_retryable_openai(exc: BaseException) -> bool:
    if _is_auth_error(exc):
        return False
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            detail = str(exc.message or "").lower()
            # FreeLLMAPI routing exhaustion — retrying just burns minutes.
            if any(
                phrase in detail
                for phrase in (
                    "all models exhausted",
                    "all models rate-limited",
                    "routing_error",
                )
            ):
                return False
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def _is_retryable_requests(exc: BaseException) -> bool:
    if isinstance(exc, requests.Timeout):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        body = (exc.response.text or "").lower()
        if status == 429 and any(
            phrase in body
            for phrase in (
                "depleted",
                "billing",
                "quota exceeded",
                "insufficient",
                "resource_exhausted",
            )
        ):
            return False
        return status in {408, 409, 429, 500, 502, 503, 504}
    if isinstance(exc, ValueError) and "depleted" in str(exc).lower():
        return False
    return False


def _format_gemini_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        message = payload.get("error", {}).get("message", response.text[:300])
    except Exception:
        message = response.text[:300]
    return f"Gemini API {response.status_code}: {message}"


def _is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


def _rate_limit_wait_seconds(exc: BaseException, attempt: int) -> int:
    """OpenRouter free tier: 20 RPM / ~50 req per day — back off generously."""
    if isinstance(exc, APIStatusError):
        detail = str(exc.message or "").lower()
        if "all models exhausted" in detail or "routing_error" in detail:
            return min(300, 90 * (attempt + 1))
    return min(180, 45 * (attempt + 1))


def _is_hard_quota_exceeded(detail: str) -> bool:
    lowered = detail.lower()
    # When free-tier is effectively disabled, Gemini returns metrics with limit: 0.
    return ("quota exceeded" in lowered and "limit: 0" in lowered) or (
        "prepayment credits are depleted" in lowered
    )


def _openai_label_messages(
    config: VisionConfig,
    *,
    base64_image: str,
    mime_type: str,
) -> list[dict]:
    if is_agentrouter(config.base_url):
        system_prompt = AGENTROUTER_SYSTEM_PROMPT
        user_prompt = AGENTROUTER_USER_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _label_with_openai(
    config: VisionConfig,
    *,
    base64_image: str,
    mime_type: str,
) -> str:
    client = create_openai_client(api_key=config.api_key, base_url=config.base_url)

    @retry(
        retry=retry_if_exception(_is_retryable_openai),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _call_api() -> str:
        response = client.chat.completions.create(
            model=config.model,
            temperature=0,
            max_tokens=1024,
            messages=_openai_label_messages(
                config, base64_image=base64_image, mime_type=mime_type
            ),
        )
        return _clean_model_output(response.choices[0].message.content or "")

    text = _call_api()
    if len(text.strip()) < MIN_LABEL_CHARS:
        raise ValueError("Vision API returned empty text (model may have hit token/rate limits)")
    return text


def _label_with_gemini(
    config: VisionConfig,
    *,
    base64_image: str,
    mime_type: str,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": USER_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": base64_image}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0},
    }

    @retry(
        retry=retry_if_exception(_is_retryable_requests),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _call_api() -> str:
        response = requests.post(
            url,
            params={"key": config.api_key},
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            detail = _format_gemini_error(response)
            if response.status_code == 429 and _is_hard_quota_exceeded(detail):
                # Stop early: all remaining images will fail the same way.
                raise SystemExit(
                    "اعتبار/سهمیهٔ Gemini شما برای درخواست‌های Vision صفر است.\n"
                    f"{detail}\n\n"
                    "لطفاً Billing/Quota را در Google AI Studio فعال کنید یا از OpenRouter/OpenAI vision استفاده کنید."
                )
            raise requests.HTTPError(detail, response=response)
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        return _clean_model_output("\n".join(text_parts))

    return _call_api()


def _label_image(config: VisionConfig, *, base64_image: str, mime_type: str) -> str:
    if config.provider == "gemini":
        return _label_with_gemini(config, base64_image=base64_image, mime_type=mime_type)
    return _label_with_openai(config, base64_image=base64_image, mime_type=mime_type)


def _verify_openai_credentials(config: VisionConfig) -> None:
    """Fail fast before labeling hundreds of images with a bad key."""
    client = create_openai_client(api_key=config.api_key, base_url=config.base_url)
    try:
        if is_agentrouter(config.base_url):
            client.chat.completions.create(
                model=config.model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
        else:
            client.models.list()
    except APIStatusError as exc:
        detail = str(exc.message or "")
        if is_insufficient_quota(detail):
            raise SystemExit(
                "شارژ AgentRouter کافی نیست.\n"
                f"{detail}\n\n"
                "از https://agentrouter.org/console/token موجودی را شارژ کن."
            ) from exc
        if exc.status_code == 404 or "无权访问模型" in detail:
            raise SystemExit(
                f"مدل {config.model} روی AgentRouter در دسترس نیست.\n"
                f"{detail}\n\n"
                "OPENAI_MODEL را در .env به مدلی که در پنل AgentRouter فعال است تغییر بده."
            ) from exc
        if exc.status_code not in {401, 403}:
            raise
        if "unauthorized client" not in detail.lower():
            raise
        base = config.base_url or OPENAI_BASE
        raise SystemExit(
            "کلید API برای این آدرس رد شد.\n\n"
            f"  base: {base}\n"
            f"  model: {config.model}\n\n"
            "AgentRouter: https://agentrouter.org/console/token\n"
            "  OPENAI_BASE_URL=https://agentrouter.org/v1\n\n"
            "تست: python scripts/verify_openai_key.py"
        ) from exc


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def label_dataset(
    *,
    config: VisionConfig,
    input_dir: Path,
    output_file: Path,
    error_log: Path,
    resume: bool,
    limit: int | None,
    sleep_sec: float,
    max_dimension: int,
) -> tuple[int, int, int]:
    images = _list_images(input_dir)
    if not images:
        logger.warning("No images found in %s", input_dir)
        return 0, 0, 0

    completed = _load_completed_paths(output_file) if resume else set()

    labeled = 0
    skipped = 0
    failed = 0
    processed = 0

    for image_path in images:
        rel_path = _relative_image_path(image_path)
        if resume and rel_path in completed:
            skipped += 1
            continue

        if limit is not None and processed >= limit:
            break

        processed += 1
        logger.info("Labeling (%d): %s", processed, rel_path)

        image_saved = False
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                base64_image, mime_type = _encode_image_base64(
                    image_path, max_dimension=max_dimension
                )
                text = _label_image(config, base64_image=base64_image, mime_type=mime_type)
                record = {"image_path": rel_path, "text": text}
                _append_jsonl(output_file, record)
                completed.add(rel_path)
                labeled += 1
                image_saved = True
                logger.info("Saved label for %s (%d chars)", image_path.name, len(text))
                break
            except Exception as exc:
                last_exc = exc
                if _is_auth_error(exc):
                    raise SystemExit(
                        "کلید API نامعتبر — لیبل‌گذاری متوقف شد.\n"
                        f"{exc}\n\n"
                        "python scripts/verify_openai_key.py را اجرا کن."
                    ) from exc
                if _is_rate_limited(exc) and attempt < 4:
                    wait = _rate_limit_wait_seconds(exc, attempt)
                    logger.warning(
                        "Rate limited on %s (attempt %d/5), sleeping %ds…",
                        image_path.name,
                        attempt + 1,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                break

        if not image_saved:
            failed += 1
            logger.error("Failed to label %s: %s", rel_path, last_exc)
            _append_jsonl(
                error_log,
                {
                    "image_path": rel_path,
                    "error": str(last_exc),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return labeled, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label images with a vision API.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--error-log", type=Path, default=ERROR_LOG_FILE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=8.0)
    parser.add_argument("--max-dimension", type=int, default=1280)
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai"),
        default=None,
        help="Vision backend: openai (FreeLLMAPI / OpenAI-compatible) or gemini.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)

    try:
        config = _load_config(args.provider)
    except SystemExit as exc:
        logger.error("%s", exc)
        return 1

    logger.info(
        "Starting auto-labeler | provider=%s model=%s base=%s input=%s output=%s resume=%s",
        config.provider,
        config.model,
        config.base_url,
        args.input_dir,
        args.output,
        args.resume,
    )

    if config.provider == "openai":
        _verify_openai_credentials(config)

    labeled, skipped, failed = label_dataset(
        config=config,
        input_dir=args.input_dir,
        output_file=args.output,
        error_log=args.error_log,
        resume=args.resume,
        limit=args.limit,
        sleep_sec=args.sleep,
        max_dimension=args.max_dimension,
    )

    logger.info(
        "Done | labeled=%d skipped=%d failed=%d output=%s",
        labeled,
        skipped,
        failed,
        args.output,
    )
    return 0 if failed == 0 or labeled > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
