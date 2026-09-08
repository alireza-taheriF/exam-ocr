"""Tests for training.eval.metrics.

Covers the three defects the rewrite fixes: WER-was-CER, macro-averaging, and
unbounded CER. Fixtures marked REAL come from dataset/ and logs/.
"""

import pytest

from training.eval.metrics import (
    accuracy_from_cer,
    cer,
    corpus_cer,
    corpus_wer,
    levenshtein,
    wer,
)

ZWNJ = "‌"


# ── Acceptance criteria from the Phase 1 brief ──────────────────────────────


def test_identical_corpus_is_zero():
    assert corpus_cer(["سلام"], ["سلام"]) == 0.0
    assert corpus_wer(["سلام دنیا"], ["سلام دنیا"]) == 0.0


def test_empty_hypothesis_is_exactly_one_not_more():
    value = corpus_cer([""], ["سلام دنیا"])
    assert value == 1.0
    assert value <= 1.0


def test_missing_hypothesis_against_real_reference_is_one():
    assert corpus_cer(["سلام دنیا"], [""]) == 1.0


# ── Clamping: the logs/eval_full_report.json pathology ──────────────────────


def test_degenerate_repetition_cannot_exceed_one():
    """REAL: logs/eval_full_report.json recorded CER 1772.38 for a 65,593-char
    hypothesis against a 39-char reference, producing a corpus mean of 225.89."""
    ref = "Subject: Date:  موضوع: تاریخ:  PRESTIGE"
    hyp = "Subject: " + "؛" * 65_000
    assert cer(ref, hyp) == 1.0
    assert corpus_cer([ref], [hyp]) <= 1.0


def test_one_degenerate_sample_cannot_dominate_the_corpus():
    refs = ["سلام"] * 9 + ["کتاب"]
    hyps = ["سلام"] * 9 + ["؛" * 5000]
    value = corpus_cer(refs, hyps)
    assert value == pytest.approx(4 / 40)  # only the bad sample's 4 chars count


# ── WER is a real WER, no longer a duplicate of CER ─────────────────────────


def test_wer_differs_from_cer():
    """The old wer() Levenshteined space-joined character strings and divided by
    character count, making it numerically identical to cer() for every input."""
    ref = "سلام دنیا خوب"
    hyp = "سلام دنیا بد"
    assert corpus_wer([ref], [hyp]) != corpus_cer([ref], [hyp])


def test_wer_counts_whole_word_substitutions():
    # 1 of 3 words wrong → 1/3, regardless of how many characters differ.
    assert corpus_wer(["الف ب ج"], ["الف ب دددددد"]) == pytest.approx(1 / 3)


def test_wer_counts_deletions_and_insertions():
    assert corpus_wer(["الف ب ج"], ["الف ب"]) == pytest.approx(1 / 3)
    assert corpus_wer(["الف ب"], ["الف ب ج"]) == pytest.approx(1 / 2)


def test_wer_is_bounded_by_one():
    assert corpus_wer(["الف"], ["ب ج د ه و"]) == 1.0


# ── Corpus aggregation, not per-sample mean ─────────────────────────────────


def test_corpus_aggregation_is_length_weighted():
    """A 1-char line scoring 1.0 and a 99-char line scoring 0.0 is corpus CER
    1/100, not the macro-average 0.5 the old implementation reported."""
    short_ref, long_ref = "ا", "ب" * 99
    refs = [short_ref, long_ref]
    hyps = ["ج", long_ref]
    assert corpus_cer(refs, hyps) == pytest.approx(1 / 100)
    macro = (cer(short_ref, "ج") + cer(long_ref, long_ref)) / 2
    assert macro == pytest.approx(0.5)
    assert corpus_cer(refs, hyps) != pytest.approx(macro)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="differ in length"):
        corpus_cer(["الف", "ب"], ["الف"])


def test_empty_corpus_is_zero():
    assert corpus_cer([], []) == 0.0
    assert corpus_wer([], []) == 0.0


# ── Normalization is applied before comparison ──────────────────────────────


def test_arabic_vs_persian_letters_score_as_equal():
    assert corpus_cer(["كتاب يك"], ["کتاب یک"]) == 0.0


def test_digit_systems_score_as_equal():
    assert corpus_cer(["۱۲۳"], ["123"]) == 0.0
    assert corpus_cer(["٤٥"], ["۴۵"]) == 0.0


def test_tashkeel_does_not_count_as_error():
    assert corpus_cer(["کتاب"], ["کِتَابُ"]) == 0.0


def test_zwnj_is_scored_not_erased():
    """The old normalizer folded ZWNJ to a space, so the model was trained on
    ZWNJ but never penalised for getting it wrong. It is now a scored character."""
    assert corpus_cer([f"می{ZWNJ}روم"], ["می روم"]) > 0.0
    assert corpus_cer([f"می{ZWNJ}روم"], [f"می{ZWNJ}روم"]) == 0.0


def test_whitespace_differences_are_normalized_away():
    assert corpus_cer(["الف ب"], ["  الف   ب  "]) == 0.0


def test_none_is_treated_as_empty():
    assert corpus_cer(["الف"], [None]) == 1.0


# ── levenshtein primitive ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "a,b,expected",
    [("", "", 0), ("abc", "abc", 0), ("", "abc", 3), ("abc", "", 3),
     ("kitten", "sitting", 3), ("flaw", "lawn", 2)],
)
def test_levenshtein_strings(a, b, expected):
    assert levenshtein(a, b) == expected


def test_levenshtein_is_symmetric():
    assert levenshtein("سلام", "سلم") == levenshtein("سلم", "سلام")


def test_levenshtein_on_token_lists():
    assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1


# ── accuracy_from_cer ───────────────────────────────────────────────────────


def test_accuracy_from_cer():
    assert accuracy_from_cer(0.0) == 100.0
    assert accuracy_from_cer(1.0) == 0.0
    assert accuracy_from_cer(0.15) == pytest.approx(85.0)
    # Bounded even if a caller passes a legacy unclamped value.
    assert accuracy_from_cer(225.89) == 0.0


# ── Single-pair helpers stay consistent with the corpus functions ────────────


def test_single_pair_helpers_match_corpus_of_one():
    ref, hyp = "سلام دنیا", "سلام دنیای"
    assert cer(ref, hyp) == corpus_cer([ref], [hyp])
    assert wer(ref, hyp) == corpus_wer([ref], [hyp])
