#!/usr/bin/env python3
"""Sync VLM_API_* in .env from vision/OpenRouter settings."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip()
    return values


def _upsert(path: Path, updates: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for key, value in updates.items():
        pat = re.compile(rf"^{re.escape(key)}=.*$", re.M)
        line = f"{key}={value}"
        text = pat.sub(line, text) if pat.search(text) else text.rstrip() + "\n" + line + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    env = _read_env(ENV)
    openai_key = env.get("OPENAI_API_KEY", "").strip()
    openrouter = env.get("OPENROUTER_API_KEY", "")
    placeholders = {"", "your_api_key_here", "changeme", "replace_me"}

    if openai_key and openai_key not in placeholders:
        model = env.get("OPENAI_MODEL") or env.get("VISION_MODEL") or "gpt-4o-mini"
        updates = {
            "VLM_API_BASE": "https://api.openai.com/v1",
            "VLM_API_KEY": openai_key,
            "VLM_MODEL": model,
        }
    elif openrouter and openrouter not in placeholders:
        updates = {
            "VLM_API_BASE": "https://openrouter.ai/api/v1",
            "VLM_API_KEY": openrouter,
            "VLM_MODEL": env.get("VISION_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free"),
        }
    elif env.get("VISION_API_KEY") and env.get("VISION_API_BASE"):
        updates = {
            "VLM_API_BASE": env["VISION_API_BASE"],
            "VLM_API_KEY": env["VISION_API_KEY"],
            "VLM_MODEL": env.get("VISION_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free"),
        }
    else:
        print("No vision/OpenRouter config found in .env")
        return 1

    _upsert(ENV, updates)
    print("Synced VLM_API_BASE, VLM_API_KEY, VLM_MODEL in .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
