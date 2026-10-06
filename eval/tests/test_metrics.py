"""Tests for sentence similarity, splitting, alignment, and partial credit."""

import pytest

from eval.metrics import (
    align_units,
    evaluate_pair,
    sentence_similarity,
    split_units,
)


def test_exact_match_similarity_one():
    assert sentence_similarity("سلام دنیا", "سلام دنیا") == 1.0


def test_partial_credit_takhir_vs_takhir_missing_alef():
    """«تاخیر» vs «تخیر» must get partial credit, not zero."""
    sim = sentence_similarity("تخیر", "تاخیر")
    assert 0.0 < sim < 1.0
    # One insertion/deletion relative to a 5-char ref → 1 - 1/5 = 0.8
    assert sim == pytest.approx(0.8)


def test_total_mismatch_can_be_zero():
    assert sentence_similarity("الف", "بجدهوز") == 0.0


def test_split_sentences_persian_punctuation():
    text = "اولی. دومی؟ سومی! چهارمی؛ پنجمی، ششمی\nهفتمی"
    units = split_units(text, mode="sentence")
    assert len(units) >= 6
    assert any("اولی" in u for u in units)
    assert any("هفتمی" in u for u in units)


def test_split_lines():
    assert split_units("الف\nب\n\nج", mode="line") == ["الف", "ب", "ج"]


def test_alignment_exact_sequence():
    refs = ["الف", "ب", "ج"]
    preds = ["الف", "ب", "ج"]
    aligned = align_units(refs, preds, page="p1")
    assert len(aligned) == 3
    assert all(u.similarity == 1.0 for u in aligned)


def test_alignment_penalizes_missing_sentence():
    refs = ["الف", "ب", "ج"]
    preds = ["الف", "ج"]  # missing «ب»
    aligned = align_units(refs, preds)
    assert len(aligned) == 3
    deletes = [u for u in aligned if u.align_tag == "delete"]
    assert len(deletes) == 1
    assert deletes[0].ref == "ب"
    assert deletes[0].similarity == 0.0
    # Remaining matches should stay high — not all shifted to zero.
    matches = [u for u in aligned if u.align_tag == "match"]
    assert any(u.similarity == 1.0 for u in matches)


def test_alignment_penalizes_extra_sentence():
    refs = ["الف", "ج"]
    preds = ["الف", "ب", "ج"]
    aligned = align_units(refs, preds)
    inserts = [u for u in aligned if u.align_tag == "insert"]
    assert len(inserts) == 1
    assert inserts[0].pred == "ب"
    assert inserts[0].similarity == 0.0


def test_evaluate_pair_partial_and_exact():
    result = evaluate_pair(
        "تخیر. سلام",
        "تاخیر. سلام",
        page="p",
        unit_mode="sentence",
    )
    assert 0.0 < result.mean_sentence_similarity < 1.0
    assert result.exact_match_rate > 0  # «سلام» exact
    assert any(u.similarity == 1.0 for u in result.units)
    assert any(0 < u.similarity < 1 for u in result.units)


def test_line_fallback_units():
    result = evaluate_pair(
        "خط یک\nخط دو",
        "خط یک\nخط دو",
        unit_mode="line",
    )
    assert result.unit_mode == "line"
    assert result.mean_sentence_similarity == 1.0
    assert len(result.units) == 2


def test_bootstrap_mean_ci_reports_n():
    from eval.metrics import bootstrap_mean_ci, aggregate_results, evaluate_pair

    ci = bootstrap_mean_ci([0.1, 0.2, 0.3, 0.4], n_boot=200, seed=0)
    assert ci["n"] == 4
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]

    pairs = [
        evaluate_pair("الف", "الف", page="p1", unit_mode="line"),
        evaluate_pair("ب", "ج", page="p2", unit_mode="line"),
    ]
    report = aggregate_results(pairs, n_boot=200)
    assert report.bootstrap_sentence["n"] == 2
    assert "ci_low" in report.to_dict()["bootstrap_sentence_similarity"]
