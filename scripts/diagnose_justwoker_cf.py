#!/usr/bin/env python3
"""Diagnose Cloudflare 403 on JustWoker POST /v1/chat/completions.

Captures the exact request/response the OpenAI SDK sends (Authorization
redacted). Does not spoof User-Agent / Origin / Referer.

    python scripts/diagnose_justwoker_cf.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "dataset" / "justwoker_cf_diagnostic.json"

_REDACT_HEADER_NAMES = frozenset({"authorization", "api-key", "x-api-key", "cookie"})


def _redact_headers(headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _REDACT_HEADER_NAMES:
            token = value.split()[-1] if value else ""
            scheme = value.split()[0] if value and " " in value else ""
            prefix = f"{scheme} " if scheme else ""
            out[key] = f"{prefix}<redacted len={len(token)} prefix={token[:4]}…>"
        else:
            out[key] = value
    return out


def _cf_bits(headers: dict[str, str], body: str) -> dict:
    lower = {k.lower(): v for k, v in headers.items()}
    title = re.search(r"<title>(.*?)</title>", body or "", re.I | re.S)
    err = re.search(r"Error code[:\s]+(\d+)", body or "", re.I)
    ray_in_body = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)", body or "", re.I)
    return {
        "cf-ray": lower.get("cf-ray"),
        "cf-mitigated": lower.get("cf-mitigated"),
        "cf-error-code": err.group(1) if err else None,
        "html_title": re.sub(r"\s+", " ", title.group(1)).strip() if title else None,
        "ray_in_body": ray_in_body.group(1).strip() if ray_in_body else None,
    }


class CaptureClient(httpx.Client):
    def __init__(self, sink: list[dict], **kwargs):
        super().__init__(**kwargs)
        self._sink = sink

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        rec = {
            "method": request.method,
            "url": str(request.url),
            "request_headers": _redact_headers(request.headers),
            "request_header_names": [k.lower() for k in request.headers.keys()],
            "has_content_type": any(k.lower() == "content-type" for k in request.headers.keys()),
            "has_user_agent": any(k.lower() == "user-agent" for k in request.headers.keys()),
            "has_origin": any(k.lower() == "origin" for k in request.headers.keys()),
            "has_referer": any(k.lower() == "referer" for k in request.headers.keys()),
            "request_body_bytes": len(request.content or b""),
        }
        response = super().send(request, **kwargs)
        body = response.text
        rec.update(
            {
                "status": response.status_code,
                "response_headers": dict(response.headers),
                "response_body": body,
                "cloudflare": _cf_bits(dict(response.headers), body),
            }
        )
        self._sink.append(rec)
        return response


def main() -> int:
    from dotenv import load_dotenv
    from openai import OpenAI

    from pipeline.openai_compat import (
        create_openai_client,
        resolve_openai_api_key,
        resolve_openai_base_url,
    )

    load_dotenv(PROJECT_ROOT / ".env")
    env = dict(os.environ)
    base = resolve_openai_base_url(env)
    key = resolve_openai_api_key(env)
    if not key:
        print("No JUSTWOKER_API_KEY / compatible key in .env")
        return 1

    captured: list[dict] = []
    # Same auth + base as create_openai_client, but with a capturing httpx
    # client so we see the exact headers the SDK puts on the wire.
    template = create_openai_client(api_key=key, base_url=base)
    http = CaptureClient(captured, follow_redirects=False)
    client = OpenAI(
        api_key=key,
        base_url=base,
        default_headers=getattr(template, "default_headers", None) or None,
        http_client=http,
    )

    print(f"base: {base}")
    print("1) GET /v1/models")
    try:
        ids = [m.id for m in client.models.list().data]
        print(f"   ok — {len(ids)} models: {', '.join(ids)}")
    except Exception as exc:
        print(f"   GET failed: {type(exc).__name__}: {exc}")

    print("2) text-only POST /v1/chat/completions × 3 (10s gap, no image)")
    for i in range(1, 4):
        print(f"   attempt {i}/3 …", flush=True)
        try:
            resp = client.chat.completions.create(
                model="gpt-5.6-sol",
                max_tokens=8,
                messages=[{"role": "user", "content": "ok"}],
            )
            text = (resp.choices[0].message.content or "").strip()
            print(f"   attempt {i}: SUCCESS text={text!r}")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            print(f"   attempt {i}: {type(exc).__name__} status={status}")
        if i < 3:
            time.sleep(10)

    report = {
        "endpoint": base,
        "note": (
            "Authorization redacted. No User-Agent/Origin/Referer spoofing. "
            "Three text-only POSTs isolate POST-general vs image-payload blocks."
        ),
        "calls": captured,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
