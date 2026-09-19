#!/usr/bin/env python3
"""Verify OpenAI / AgentRouter API before labeling hundreds of images."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.openai_compat import (
    create_openai_client,
    format_agentrouter_error,
    is_agentrouter,
    is_justwoker,
    is_insufficient_quota,
    resolve_openai_api_key,
    resolve_openai_base_url,
)


def _key_hint(key: str) -> str:
    if key.startswith("sk-proj-"):
        return "فرمت کلید رسمی OpenAI (sk-proj-)"
    base = resolve_openai_base_url(dict(os.environ))
    if is_justwoker(base):
        return "کلید JustWoker"
    if is_agentrouter(base):
        return "کلید AgentRouter"
    if key.startswith("sk-"):
        return f"فرمت sk- (طول {len(key)})"
    return "فرمت کلید ناشناخته"


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = resolve_openai_api_key(os.environ)
    base_url = resolve_openai_base_url(os.environ)
    model = (
        os.environ.get("OPENAI_MODEL", "").strip()
        or os.environ.get("VISION_MODEL", "").strip()
        or "gpt-4o-mini"
    )

    if not api_key:
        print("OPENAI_API_KEY در .env یافت نشد.")
        return 1

    print(f"base:  {base_url}")
    print(f"model: {model}")
    print(f"key:   {_key_hint(api_key)}")
    if is_agentrouter(base_url):
        print("mode:  AgentRouter (Codex headers)")
    print()

    client = create_openai_client(api_key=api_key, base_url=base_url)
    try:
        if is_agentrouter(base_url):
            client.chat.completions.create(
                model=model,
                max_tokens=8,
                messages=[{"role": "user", "content": "ping"}],
            )
        else:
            client.models.list()
        print("✓ کلید معتبر است — اتصال OK")
    except APIStatusError as exc:
        detail = str(exc.message or "")
        if exc.status_code in {401, 403} and "无权访问模型" not in detail and "unauthorized client" in detail.lower():
            print("✗ خطای احراز هویت:", exc.message)
            if is_agentrouter(base_url):
                print("  کلید را از https://agentrouter.org/console/token بررسی کن.")
            return 1
        if exc.status_code in {403, 404} or "无权访问模型" in detail or is_insufficient_quota(detail):
            print(f"✗ {format_agentrouter_error(detail) if is_agentrouter(base_url) else detail[:200]}")
            if is_agentrouter(base_url) and not is_insufficient_quota(detail):
                print("  OPENAI_MODEL را در .env به مدلی که در پنل AgentRouter فعال است تغییر بده.")
            return 1
        print(f"✗ خطای API: {exc.status_code} — {exc.message}")
        return 1
    except Exception as exc:
        print(f"✗ خطای اتصال: {exc}")
        return 1

    sample = ROOT / "dataset" / "raw_images"
    images = sorted(
        p
        for p in sample.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        return 0

    img = images[0]
    from io import BytesIO
    from PIL import Image

    with Image.open(img) as pil:
        pil = pil.convert("RGB")
        w, h = pil.size
        longest = max(w, h)
        if longest > 1280:
            scale = 1280 / longest
            pil = pil.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=92)
    data = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg"
    try:
        from pipeline.openai_compat import AGENTROUTER_SYSTEM_PROMPT, AGENTROUTER_USER_PROMPT
        vision_messages = (
            [
                {"role": "system", "content": AGENTROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                        {"type": "text", "text": AGENTROUTER_USER_PROMPT},
                    ],
                },
            ]
            if is_agentrouter(base_url)
            else [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract one Persian word from the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        },
                    ],
                }
            ]
        )
        client.chat.completions.create(
            model=model,
            max_tokens=32,
            messages=vision_messages,
        )
        print(f"✓ مدل vision ({model}) روی {img.name} پاسخ داد")
    except APIStatusError as exc:
        detail = str(exc.message or "")
        if is_insufficient_quota(detail):
            print(f"✗ {format_agentrouter_error(detail)}")
        else:
            print(f"✗ مدل {model} خطا داد: {exc.status_code} — {exc.message}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
