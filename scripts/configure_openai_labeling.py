#!/usr/bin/env python3
"""
Configure .env for OpenAI-compatible vision labeling.

Official OpenAI:
    OPENAI_API_KEY=sk-proj-...
    python scripts/configure_openai_labeling.py

AgentRouter (https://agentrouter.org/console/token):
    OPENAI_API_KEY=sk-...
    OPENAI_BASE_URL=https://agentrouter.org/v1
    OPENAI_MODEL=gpt-5
    python scripts/configure_openai_labeling.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
AGENTROUTER_BASE = "https://agentrouter.org/v1"
JUSTWOKER_BASE = "https://api.justwoker.icu/v1"

PLACEHOLDERS = {"", "your_api_key_here", "changeme", "replace_me"}


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            values[k.strip()] = v.strip()
    return values


def _rewrite_env(path: Path, updates: dict[str, str]) -> None:
    lines: list[str] = []
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key = line.partition("=")[0].strip()
                if key in updates:
                    if key not in seen:
                        lines.append(f"{key}={updates[key]}")
                        seen.add(key)
                    continue
            lines.append(line)

    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}={value}")
            seen.add(key)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _resolve_base(env: dict[str, str]) -> str:
    justwoker = env.get("JUSTWOKER_BASE_URL", "").strip()
    if justwoker:
        return justwoker.rstrip("/")
    if env.get("JUSTWOKER_API_KEY", "").strip():
        return JUSTWOKER_BASE
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "VISION_API_BASE",
        "VLM_API_BASE",
    ):
        value = env.get(name, "").strip()
        if value:
            return value.rstrip("/")
    return "https://api.openai.com/v1"


def _default_model(env: dict[str, str], base: str) -> str:
    model = env.get("OPENAI_MODEL") or env.get("VISION_MODEL") or ""
    if model:
        return model
    if "justwoker.icu" in base:
        return "gpt-5.6-sol"
    if "agentrouter.org" in base:
        # gpt-5.5 often hits quota; claude-opus-4-6 works on default group.
        return "claude-opus-4-6"
    return "gpt-4o-mini"


def main() -> int:
    env = _read_env(ENV)
    api_key = (
        env.get("JUSTWOKER_API_KEY", "").strip()
        or env.get("OPENAI_API_KEY", "").strip()
        or env.get("AGENT_ROUTER_TOKEN", "").strip()
    )
    if not api_key or api_key in PLACEHOLDERS:
        print(
            "JUSTWOKER_API_KEY / OPENAI_API_KEY not found in .env\n\n"
            "JustWoker:   set JUSTWOKER_API_KEY (do not reuse the AgentRouter key)\n"
            "AgentRouter: https://agentrouter.org/console/token\n"
            "OpenAI:      https://platform.openai.com/api-keys\n"
        )
        return 1

    base = _resolve_base(env)
    using_justwoker = bool(env.get("JUSTWOKER_API_KEY", "").strip()) or "justwoker.icu" in base
    if (
        not using_justwoker
        and not env.get("OPENAI_BASE_URL")
        and api_key.startswith("sk-")
        and not api_key.startswith("sk-proj-")
    ):
        base = AGENTROUTER_BASE
    model = _default_model(env, base)

    updates = {
        "OPENAI_MODEL": model,
        "OPENAI_BASE_URL": base,
        "VISION_API_BASE": base,
        "VISION_API_KEY": api_key,
        "VISION_MODEL": model,
        "VLM_API_BASE": base,
        "VLM_API_KEY": api_key,
        "VLM_MODEL": model,
        "LABEL_PROVIDER": "openai",
        "LABEL_OPENAI_ONLY": "true",
        "LABEL_DIRECT_OPENROUTER": "false",
    }
    if using_justwoker:
        updates["JUSTWOKER_API_KEY"] = api_key
        updates["JUSTWOKER_BASE_URL"] = JUSTWOKER_BASE
        # Leave OPENAI_API_KEY untouched so the AgentRouter credential stays
        # distinct from this provider.
    else:
        updates["OPENAI_API_KEY"] = api_key
    _rewrite_env(ENV, updates)

    print("Configured vision API for labeling and OCR pipeline:")
    print(f"  base:  {base}")
    print(f"  model: {model}")
    if "justwoker.icu" in base:
        print("  provider: JustWoker (OpenAI-compatible chat completions)")
    elif "agentrouter.org" in base:
        print("  provider: AgentRouter (Codex client headers)")
    print()
    print("Verify:")
    print("  python scripts/verify_openai_key.py")
    print()
    print("Label one page first:")
    print("  python auto_labeler.py --resume --sleep 2 --limit 1")

    verify = ROOT / "scripts" / "verify_openai_key.py"
    if verify.exists():
        print()
        result = subprocess.run([sys.executable, str(verify)], cwd=ROOT)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
