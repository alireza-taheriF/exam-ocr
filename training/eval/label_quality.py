"""Heuristics for detecting bad auto-labels (hallucinations, empty, etc.)."""

from __future__ import annotations

import re
from collections import Counter

from training.eval.metrics import normalize_text

_PERSIAN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def persian_char_ratio(text: str) -> float:
    stripped = normalize_text(text)
    if not stripped:
        return 0.0
    persian = len(_PERSIAN_RE.findall(stripped))
    return persian / len(stripped.replace(" ", ""))


def is_valid_label(
    text: str,
    *,
    max_chars: int = 2000,
    min_chars: int = 2,
    min_persian_ratio: float = 0.15,
    max_word_repeat_ratio: float = 0.4,
) -> bool:
    """Return False for empty, hallucinated, or obviously wrong labels."""
    stripped = normalize_text(text)
    if len(stripped) < min_chars:
        return False
    if len(stripped) > max_chars:
        return False

    words = stripped.split()
    if len(words) >= 10:
        top_count = Counter(words).most_common(1)[0][1]
        if top_count / len(words) > max_word_repeat_ratio:
            return False

    # Allow math-heavy pages with low Persian ratio if short
    if len(stripped) > 30 and persian_char_ratio(stripped) < min_persian_ratio:
        return False

    return True


def rejection_reason(text: str) -> str | None:
    if is_valid_label(text):
        return None
    stripped = normalize_text(text)
    if len(stripped) < 2:
        return "too_short"
    if len(stripped) > 2000:
        return "too_long"
    words = stripped.split()
    if len(words) >= 10:
        top_count = Counter(words).most_common(1)[0][1]
        if top_count / len(words) > 0.4:
            return "repetition_hallucination"
    if len(stripped) > 30 and persian_char_ratio(stripped) < 0.15:
        return "low_persian_ratio"
    return "invalid"
