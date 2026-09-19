"""OpenAI-compatible client helpers (JustWoker, AgentRouter, etc.)."""

from __future__ import annotations

import os

from openai import OpenAI

JUSTWOKER_HOST = "justwoker.icu"
JUSTWOKER_DEFAULT_BASE = "https://api.justwoker.icu/v1"

AGENTROUTER_HOST = "agentrouter.org"

# AgentRouter only accepts requests that look like the Codex CLI.
# https://agentrouter.org — Codex config uses base_url https://agentrouter.org/v1
CODEX_CLIENT_HEADERS = {
    "Originator": "codex_cli_rs",
    "User-Agent": "codex_cli_rs/0.101.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464",
    "Version": "0.101.0",
}

# AgentRouter edge filter blocks Persian system prompts (content-blocked).
AGENTROUTER_SYSTEM_PROMPT = (
    "You are an OCR/HTR engine for Persian (Farsi). "
    "Read handwritten or printed text in the image exactly as written. "
    "Output only the extracted text with no extra commentary."
)
AGENTROUTER_USER_PROMPT = "Extract all Persian text from this image exactly as written."


def is_agentrouter(base_url: str | None) -> bool:
    return bool(base_url and AGENTROUTER_HOST in base_url.lower())


def is_justwoker(base_url: str | None) -> bool:
    return bool(base_url and JUSTWOKER_HOST in base_url.lower())


def resolve_openai_base_url(environ: dict[str, str]) -> str:
    justwoker_base = environ.get("JUSTWOKER_BASE_URL", "").strip()
    if justwoker_base:
        return justwoker_base.rstrip("/")
    # Dedicated JustWoker key wins over leftover AgentRouter OPENAI_BASE_URL.
    if environ.get("JUSTWOKER_API_KEY", "").strip():
        return JUSTWOKER_DEFAULT_BASE
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "VISION_API_BASE",
        "VLM_API_BASE",
    ):
        value = environ.get(name, "").strip()
        if value:
            return value.rstrip("/")
    return "https://api.openai.com/v1"


def resolve_openai_api_key(environ: dict[str, str]) -> str:
    for name in (
        "JUSTWOKER_API_KEY",
        "OPENAI_API_KEY",
        "AGENT_ROUTER_TOKEN",
        "VISION_API_KEY",
        "VLM_API_KEY",
    ):
        value = environ.get(name, "").strip()
        if value:
            return value
    return ""


def is_insufficient_quota(detail: str) -> bool:
    lowered = detail.lower()
    return (
        "insufficient_user_quota" in lowered
        or "额度不足" in detail
        or "insufficient quota" in lowered
        or "prepayment credits are depleted" in lowered
    )


def format_agentrouter_error(detail: str) -> str:
    if is_insufficient_quota(detail):
        return (
            "سقف توکن API کافی نیست (ممکن است با موجودی حساب $125 فرق داشته باشد).\n"
            "در Console → Token/API Key:\n"
            "  • توکن جدید بساز\n"
            "  • یا «无限额度 / Unlimited» را فعال کن\n"
            "  • یا Remaining Quota توکن را از موجودی حساب شارژ کن\n"
            "سپس OPENAI_API_KEY در .env را با توکن جدید عوض کن.\n"
            "https://agentrouter.org/console/token"
        )
    if "无权访问模型" in detail:
        return "توکن اجازه دسترسی به این مدل را ندارد — OPENAI_MODEL را عوض کن."
    if "content-blocked" in detail.lower():
        return (
            "AgentRouter محتوای تصویر را block کرد (content-blocked).\n"
            "مدل دیگر امتحان کن یا از پشتیبانی AgentRouter بپرس."
        )
    if "无可用渠道" in detail or "no available channel" in detail.lower():
        return (
            "این مدل در گروه default روی AgentRouter کانال ندارد.\n"
            "مدل‌های موجود معمولاً: claude-opus-4-6, glm-5.2, gpt-5.5\n"
            "python scripts/verify_openai_key.py را اجرا کن."
        )
    return detail[:300]


def create_openai_client(*, api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    env = dict(os.environ)
    key = (api_key if api_key is not None else resolve_openai_api_key(env)).strip()
    normalized = (base_url if base_url is not None else resolve_openai_base_url(env)).rstrip("/")
    if not normalized:
        normalized = "https://api.openai.com/v1"
    kwargs: dict = {"api_key": key, "base_url": normalized}
    # JustWoker is OpenAI chat-completions compatible; Codex headers are
    # AgentRouter-only and would be the wrong client fingerprint here.
    if is_agentrouter(normalized):
        kwargs["default_headers"] = dict(CODEX_CLIENT_HEADERS)
    return OpenAI(**kwargs)
