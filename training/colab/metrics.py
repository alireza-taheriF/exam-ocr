"""CER/WER metrics (same as training/eval/metrics.py) — used in Colab without full repo."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u064a", "\u06cc")
    normalized = normalized.replace("\u0643", "\u06a9")
    normalized = normalized.replace("\u200c", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein_distance(ref, hyp) / len(ref)


def accuracy_from_cer(cer_value: float) -> float:
    return max(0.0, (1.0 - cer_value) * 100.0)
