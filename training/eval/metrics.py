"""
Character and word error rate metrics for Persian OCR evaluation.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Normalize Persian/Arabic text for fair CER/WER comparison."""
    if not text:
        return ""
    # Unicode compatibility normalization
    normalized = unicodedata.normalize("NFKC", text)
    # Unify Arabic/Persian variants
    normalized = normalized.replace("\u064a", "\u06cc")  # ي → ی
    normalized = normalized.replace("\u0643", "\u06a9")  # ك → ک
    normalized = normalized.replace("\u200c", " ")  # ZWNJ → space for comparison
    # Collapse whitespace
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
    """
    Character Error Rate in [0, inf).
    Returns 0.0 for identical strings; 1.0 means 100% character errors.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate based on whitespace tokenization."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_distance(" ".join(ref_words), " ".join(hyp_words)) / len(
        " ".join(ref_words)
    )


def accuracy_from_cer(cer_value: float) -> float:
    """Convert CER to character accuracy percentage."""
    return max(0.0, (1.0 - cer_value) * 100.0)
