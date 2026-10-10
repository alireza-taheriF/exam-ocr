"""Crops that must never enter training or tuning (see .claude/skills/ocr-eval-gate).

One place to ask "is this crop sealed?", used by the calibration pool, the
LoRA data builder and the Colab bundler:

- dataset/gold.jsonl                 — the scoring gold set
- dataset/test_v3.jsonl              — the test split gold is drawn from
- eval/new_gold/manifest.jsonl       — the sealed 300-line set on branch
  cursor/report-accuracy-graphs; read from the working tree if present,
  otherwise from that branch via git. It spans train/val/test.

Matching is by crop file name (the sha1-derived basename in dataset/lines_v3),
so a path written differently still matches.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_GOLD_REF = "cursor/report-accuracy-graphs:eval/new_gold/manifest.jsonl"


def _names_from_jsonl(text: str) -> set[str]:
    names = set()
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            for key in ("image_path", "crop_path"):
                if row.get(key):
                    names.add(Path(row[key]).name)
    return names


def new_gold_names() -> set[str]:
    local = PROJECT_ROOT / "eval" / "new_gold" / "manifest.jsonl"
    if local.exists():
        return _names_from_jsonl(local.read_text(encoding="utf-8"))
    try:
        out = subprocess.run(["git", "show", NEW_GOLD_REF], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return _names_from_jsonl(out)


def sealed_crop_names(include_test: bool = True) -> set[str]:
    names = new_gold_names()
    for rel in ["dataset/gold.jsonl"] + (["dataset/test_v3.jsonl"] if include_test else []):
        path = PROJECT_ROOT / rel
        if path.exists():
            names |= _names_from_jsonl(path.read_text(encoding="utf-8"))
    return names
