from learning.correction_memory import apply_rules, build_memory, extract_substitutions
from learning.label_trust import QUARANTINED, REJECTED, TRUSTED, assess


def _row(pred, final, author, course="c", consent=1, holdout=0, conf=0.5):
    return {"predicted_text": pred, "final_text": final, "author": author, "course": course,
            "consent": consent, "holdout": holdout, "confidence": conf}


def test_extract_word_and_phrase_substitutions():
    assert extract_substitutions("به سست چپ", "به سمت چپ") == [("سست", "سمت")]
    assert extract_substitutions("معا دل۹ اساتی", "معادله اساسی") == [("معا دل۹ اساتی", "معادله اساسی")]
    assert extract_substitutions("يک", "یک") == []  # normalization-only change is not a rule


def test_rule_needs_independent_authors_and_precision():
    rows = [_row("به سست چپ", "به سمت چپ", "a"), _row("سست است", "سمت است", "a")]
    assert build_memory(rows, min_authors=2, min_precision=0.9).rules == []
    rows.append(_row("سست شد", "سمت شد", "b"))
    mem = build_memory(rows, min_authors=2, min_precision=0.9)
    assert [(r.src, r.dst, r.scope) for r in mem.rules if r.scope == "*"] == [("سست", "سمت", "*")]
    # A reviewed line that kept "سست" lowers precision below the bar.
    rows.append(_row("خیلی سست", "خیلی سست", "c"))
    assert build_memory(rows, min_authors=2, min_precision=0.9).rules == []


def test_holdout_and_no_consent_rows_are_ignored():
    rows = [_row("سست", "سمت", "a", holdout=1), _row("سست", "سمت", "b", consent=0)]
    assert build_memory(rows, min_authors=1, min_precision=0.5).rules == []


def test_apply_rules_longest_phrase_first():
    rows = [_row("معا دل۹ اساتی", "معادله اساسی", "a"), _row("دل۹", "دلیل", "a")]
    mem = build_memory(rows, min_authors=1, min_precision=0.5)
    text, applied = apply_rules("این معا دل۹ اساتی است", mem.rules_for("c"))
    assert text == "این معادله اساسی است"
    assert len(applied) == 1


def test_trust():
    assert assess("به سست چپ", "به سمت چپ", 0.9).status == TRUSTED
    assert assess("پاسخ من الف است", "جواب درست گزینه ج می‌باشد", 0.95).status == QUARANTINED
    assert assess("متن", "", 0.5).status == QUARANTINED
    assert assess("متن", "||||", 0.5).status == REJECTED
