"""Model registry helpers: which adapter is live, activation and rollback.

``models/registry.json`` is written by ``scripts/eval_gate.py --register``:
``champion`` is the live entry and ``history`` every scored version with status
``champion`` / ``retired`` / ``rejected``. The OCR engine loads the champion's
adapter unless ``QWEN_ADAPTER_PATH`` overrides it, so promoting a model is a
registry change, and rollback is re-activating the previous entry.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def registry_path() -> Path:
    return Path(os.environ.get("MODEL_REGISTRY", PROJECT_ROOT / "models" / "registry.json"))


def load() -> dict:
    path = registry_path()
    if not path.exists():
        return {"champion": None, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save(registry: dict) -> None:
    registry_path().write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def champion_adapter() -> str:
    """Adapter dir of the live model ('' = base model). A missing dir is an error, not a silent fallback."""
    champ = load().get("champion") or {}
    adapter = champ.get("adapter") or ""
    if not adapter:
        return ""
    path = Path(adapter)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"registry champion {champ.get('label')!r} points at missing adapter {path}")
    return str(path)


def activate(label: str, *, reason: str) -> dict:
    """Make a previously scored entry the champion (used for rollback). Never activates a rejected entry."""
    registry = load()
    entry = next((e for e in reversed(registry["history"]) if e.get("label") == label), None)
    if entry is None:
        raise KeyError(f"{label!r} is not in the registry")
    if entry.get("status") == "rejected":
        raise ValueError(f"{label!r} was rejected by the eval gate; it cannot be activated")
    current = registry.get("champion")
    if current and current.get("label") == label:
        return current
    if current:
        registry["history"].append({**current, "status": "retired", "retired_at": time.time()})
    registry["champion"] = {**entry, "status": "champion", "activated_at": time.time(), "activation_reason": reason}
    registry["history"].append(registry["champion"])
    save(registry)
    return registry["champion"]


def previous_champion() -> str | None:
    """Label of the most recent champion before the current one, for one-command rollback."""
    registry = load()
    current = (registry.get("champion") or {}).get("label")
    for e in reversed(registry["history"]):
        if e.get("status") in {"champion", "retired"} and e.get("label") != current:
            return e["label"]
    return None
