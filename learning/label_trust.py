"""Trust assessment for student corrections (see `.claude/skills/feedback-learning`).

A student's edit is a noisy label: on an exam answer sheet the student may also
rewrite an answer instead of fixing the OCR. Corrections that look like content
rewrites are quarantined — kept, shown to the instructor, but never used for
learning until a human reviews them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.normalizer import normalize_text, vlm_artifact_reason
from training.eval.metrics import levenshtein

TRUSTED = "trusted"
QUARANTINED = "quarantined"
REJECTED = "rejected"

# A confident prediction that the student changed by more than this fraction is
# more likely a rewrite than an OCR fix.
CONFIDENT = 0.80
REWRITE_ON_CONFIDENT = 0.60
NEAR_TOTAL_REWRITE = 0.90


@dataclass
class Trust:
    status: str
    edit_ratio: float
    reasons: list[str] = field(default_factory=list)


def edit_ratio(predicted: str, final: str) -> float:
    """Normalized edit distance in [0, 1] between what was shown and what was saved."""
    a, b = normalize_text(predicted or ""), normalize_text(final or "")
    longest = max(len(a), len(b))
    return levenshtein(a, b) / longest if longest else 0.0


def assess(predicted: str, final: str, confidence: float | None) -> Trust:
    ratio = edit_ratio(predicted, final)
    if not normalize_text(final or ""):
        return Trust(QUARANTINED, ratio, ["cleared"])
    artifact = vlm_artifact_reason(final)
    if artifact:
        return Trust(REJECTED, ratio, [f"artifact:{artifact}"])

    reasons: list[str] = []
    if ratio > NEAR_TOTAL_REWRITE and normalize_text(predicted or ""):
        reasons.append("near_total_rewrite")
    if confidence is not None and confidence >= CONFIDENT and ratio > REWRITE_ON_CONFIDENT:
        reasons.append("rewrite_on_confident_line")
    return Trust(QUARANTINED if reasons else TRUSTED, ratio, reasons)
