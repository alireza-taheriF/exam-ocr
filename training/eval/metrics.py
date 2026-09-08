"""Corpus-level CER/WER for Persian OCR evaluation.

Three defects in the previous implementation are fixed here:

1. **`wer()` was not WER.** It ran Levenshtein over the space-joined *character*
   strings and divided by *character* count, making it numerically identical to
   `cer()` for every input. Every WER figure in `logs/` is a duplicate of CER.
   The corpus WER below edit-distances the *token lists* and divides by the
   reference word count.
2. **Scores were macro-averaged per sample.** `sum(cer_i)/n` over-weights short
   lines, and this corpus is short-line-heavy (median 34 chars, max 1,023). The
   standard aggregation — and the default here — is `Σ distance / Σ len(ref)`.
3. **CER was unbounded above.** With no ceiling, one degenerate hypothesis
   (65,593 chars against a 39-char reference) produced a per-sample CER of
   1772.38 and a reported corpus mean of 225.89. Rates are clamped to 1.0.

Normalization is delegated to `pipeline.normalizer`, so scores reflect
normalized Persian text and ZWNJ is no longer silently folded to a space.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):  # allow `python training/eval/metrics.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.normalizer import normalize_text as _normalize

__all__ = [
    "normalize_text",
    "corpus_cer",
    "corpus_wer",
    "cer",
    "wer",
    "accuracy_from_cer",
    "levenshtein",
]


def normalize_text(text: str) -> str:
    """Back-compat shim. Prefer importing from `pipeline.normalizer`."""
    return _normalize(text)


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance over any two sequences — strings for CER, token lists for WER."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):  # keep the inner row short
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _aggregate(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    tokenize: bool,
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses differ in length: "
            f"{len(references)} vs {len(hypotheses)}"
        )

    total_distance = 0
    total_length = 0
    for ref_raw, hyp_raw in zip(references, hypotheses):
        ref = _normalize(ref_raw or "")
        hyp = _normalize(hyp_raw or "")
        if tokenize:
            ref_seq: Sequence = ref.split()
            hyp_seq: Sequence = hyp.split()
        else:
            ref_seq, hyp_seq = ref, hyp

        if not ref_seq:
            # Empty reference with a non-empty hypothesis is 100% insertion error.
            # Using len(hyp) as that sample's denominator scores it 1.0 rather
            # than dividing by zero or silently dropping it.
            if hyp_seq:
                total_distance += len(hyp_seq)
                total_length += len(hyp_seq)
            continue
        # Clamp per sample: without this, a degenerate hypothesis far longer than
        # its reference contributes distance > len(ref) and drags the corpus rate
        # above 1.0.
        total_distance += min(levenshtein(ref_seq, hyp_seq), len(ref_seq))
        total_length += len(ref_seq)

    if total_length == 0:
        return 0.0
    return total_distance / total_length


def corpus_cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Character Error Rate in [0.0, 1.0].

    `Σ edit_distance(hyp_i, ref_i) / Σ len(ref_i)` over the whole corpus — not a
    per-line average. Both sides are normalized via `pipeline.normalizer`.
    """
    return _aggregate(references, hypotheses, tokenize=False)


def corpus_wer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Corpus-level Word Error Rate in [0.0, 1.0], whitespace-tokenized."""
    return _aggregate(references, hypotheses, tokenize=True)


def cer(reference: str, hypothesis: str) -> float:
    """Single-pair CER, clamped to [0.0, 1.0]. For diagnostics only — report
    `corpus_cer` over the full set, never the mean of these."""
    return corpus_cer([reference], [hypothesis])


def wer(reference: str, hypothesis: str) -> float:
    """Single-pair WER, clamped to [0.0, 1.0]. Diagnostics only."""
    return corpus_wer([reference], [hypothesis])


def accuracy_from_cer(cer_value: float) -> float:
    """Character accuracy percentage. Meaningful now that CER is bounded by 1.0."""
    return max(0.0, min(1.0, 1.0 - cer_value)) * 100.0
