"""Tests for eval.normalize."""

from eval.normalize import normalize_text


def test_yeh_kaf_unified():
    assert normalize_text("يك") == "یک"


def test_digits_to_persian():
    assert normalize_text("123") == "۱۲۳"
    assert normalize_text("١٢٣") == "۱۲۳"


def test_strip_tatweel_and_tashkeel():
    assert "ـ" not in normalize_text("سـلام")
    assert normalize_text("سَلام") == "سلام"


def test_zwnj_stripped_consistently():
    assert normalize_text("می‌روم") == normalize_text("می\u200cروم")
    assert "\u200c" not in normalize_text("می‌روم")


def test_whitespace_collapsed():
    assert normalize_text("سلام   دنیا\n\tخوب") == "سلام دنیا خوب"
