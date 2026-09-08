"""Tests for pipeline.normalizer.

Regression fixtures marked REAL are verbatim strings pulled from the existing
`dataset/train.jsonl` / `dataset/lines.jsonl`, so a future change to the
normalizer that breaks real data fails here rather than in training.
"""

import pytest

from pipeline.normalizer import (
    ZWNJ,
    is_vlm_artifact,
    normalize_text,
    persian_char_ratio,
    vlm_artifact_reason,
)

# ── ZWNJ preservation — the whole point of rewriting the old normalizer ──────


def test_zwnj_survives_normalization():
    assert ZWNJ in normalize_text(f"می{ZWNJ}روم")


def test_zwnj_is_not_collapsed_into_space():
    assert normalize_text(f"می{ZWNJ}روم") == f"می{ZWNJ}روم"


def test_zwnj_distinguishes_two_spellings():
    joined = normalize_text(f"می{ZWNJ}روم")
    spaced = normalize_text("می روم")
    assert joined != spaced, "ZWNJ must remain a distinguishing character"


def test_zwnj_survives_real_label():
    # REAL: dataset/train.jsonl — one of 1,429 rows containing a ZWNJ
    raw = f"ثبتی انجام نمی{ZWNJ}شود"
    out = normalize_text(raw)
    assert out == raw
    assert out.count(ZWNJ) == 1


# ── Digit unification ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    ["1234567890", "١٢٣٤٥٦٧٨٩٠", "۱۲۳۴۵۶۷۸۹۰"],
)
def test_all_three_digit_systems_unify_to_persian(raw):
    assert normalize_text(raw, digits="fa") == "۱۲۳۴۵۶۷۸۹۰"


@pytest.mark.parametrize(
    "raw",
    ["1234567890", "١٢٣٤٥٦٧٨٩٠", "۱۲۳۴۵۶۷۸۹۰"],
)
def test_all_three_digit_systems_unify_to_ascii(raw):
    assert normalize_text(raw, digits="ascii") == "1234567890"


def test_digit_unification_round_trip():
    fa = normalize_text("42", digits="fa")
    assert fa == "۴۲"
    assert normalize_text(fa, digits="ascii") == "42"
    assert normalize_text(normalize_text(fa, digits="ascii"), digits="fa") == fa


def test_mixed_digit_systems_in_one_string():
    assert normalize_text("قیمت 1200 و ٤٥ و ۶۷") == "قیمت ۱۲۰۰ و ۴۵ و ۶۷"


def test_digits_keep_leaves_all_three_alone():
    raw = "1 ١ ۱"
    assert normalize_text(raw, digits="keep") == raw


def test_digit_unification_on_real_accounting_label():
    # REAL: dataset/lines.jsonl 05ea3f9df9cd.jpg
    assert normalize_text("موجودی کالای شعبه ۸۵۰/۰۰۰") == "موجودی کالای شعبه ۸۵۰/۰۰۰"


# ── Tashkeel and tatweel ────────────────────────────────────────────────────


def test_tashkeel_stripped():
    assert normalize_text("کِتَابُ") == "کتاب"


def test_all_harakat_stripped():
    # fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun
    assert normalize_text("بًبٌبٍبَبُبِبّبْ") == "بببببببب"


def test_tatweel_removed():
    assert normalize_text("بســـیار") == "بسیار"


def test_tashkeel_can_be_kept():
    assert "َ" in normalize_text("کَتاب", strip_tashkeel=False)


# ── Letter folding ──────────────────────────────────────────────────────────


def test_arabic_kaf_and_yeh_fold_to_persian():
    assert normalize_text("كتاب يك") == "کتاب یک"


def test_alef_maksura_folds_to_persian_yeh():
    assert normalize_text("علیى") == "علیی"


def test_hamza_alef_variants_fold_but_madda_is_preserved():
    # آ is a distinct Persian letter (آب) and must NOT fold to ا.
    assert normalize_text("أحمد إسلام ٱلله آب") == "احمد اسلام الله آب"


def test_yeh_and_waw_with_hamza_are_preserved():
    # مسئول / مؤسسه are correct Persian spellings.
    assert normalize_text("مسئول مؤسسه") == "مسئول مؤسسه"


def test_teh_marbuta_and_heh_with_yeh_above():
    assert normalize_text("مکة") == "مکه"
    assert normalize_text("خانۀ") == "خانه"


def test_teh_marbuta_target_is_configurable():
    assert normalize_text("قوة", teh_marbuta="ت") == "قوت"


# ── Whitespace and control characters ───────────────────────────────────────


def test_double_space_collapsed():
    # 8.57% of the old train.jsonl rows contained a double space.
    assert normalize_text("الف   ب") == "الف ب"


def test_leading_and_trailing_whitespace_stripped():
    assert normalize_text("  سلام  ") == "سلام"


def test_exotic_spaces_collapsed():
    # NBSP, ideographic space, thin space, ZWSP
    assert normalize_text("الف 　 ​ب") == "الف ب"


def test_directional_marks_removed_but_zwnj_kept():
    assert normalize_text(f"الف‎‏{ZWNJ}ب") == f"الف{ZWNJ}ب"


def test_empty_and_none_like_input():
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalization_is_idempotent():
    raw = f"كتاب  1٢۳ بســیار نمی{ZWNJ}شود  "
    once = normalize_text(raw)
    assert normalize_text(once) == once


# ── Documented lossy behaviour (NFKC) ───────────────────────────────────────


def test_arabic_presentation_forms_are_folded():
    assert normalize_text("ﻻ") == "لا"


def test_superscripts_flatten_known_tradeoff():
    """NFKC folds ⁴ → 4, so exponent notation loses its superscript.

    Pinned deliberately: math-heavy pages in this corpus use Unicode super- and
    subscripts, and this is the one place normalization discards information.
    Revisit with a targeted presentation-form fold + NFC if math CER matters.
    """
    assert normalize_text("x⁴ + y₂") == "x۴ + y۲"


# ── VLM artifact detection ──────────────────────────────────────────────────


def test_markdown_table_rule_is_artifact():
    # REAL: appears 18× as a standalone training label in dataset/train.jsonl
    assert is_vlm_artifact("|---|---|---|")
    assert vlm_artifact_reason("|---|---|---|") == "markdown_table_rule"


def test_markdown_table_row_is_artifact():
    # REAL: dataset/lines.jsonl c931871ef2ea.jpg — label paired with a crop that
    # actually shows a handwritten arithmetic line.
    row = "| مانده بستانکار | مانده بدهکار | |"
    assert vlm_artifact_reason(row) == "markdown_table_row"


def test_real_mispaired_table_label_is_rejected():
    # REAL: dataset/lines.jsonl 38ff33bafa5d.jpg
    label = "| ۳. خرید نسیه توسط شعبه | خرید کالا ۲۰۰/۰۰۰/۰۰۰ | ثبتی انجام نمی‌شود |"
    assert is_vlm_artifact(label)


@pytest.mark.parametrize("raw", ["🎓 دانشگاه", "🐝", "😐 سلام"])
def test_emoji_is_artifact(raw):
    assert vlm_artifact_reason(raw) == "emoji_or_dingbat"


@pytest.mark.parametrize("raw", ["① اول", "⑧ ...۱۲۰۰", "❶ الف"])
def test_circled_glyph_is_artifact(raw):
    assert vlm_artifact_reason(raw) == "circled_glyph"


def test_punctuation_only_is_artifact():
    assert vlm_artifact_reason("....") == "punctuation_only"
    assert vlm_artifact_reason("---") == "punctuation_only"


def test_empty_is_artifact():
    assert vlm_artifact_reason("") == "empty"
    assert vlm_artifact_reason("   ") == "empty"


def test_greek_heavy_line_is_artifact():
    assert vlm_artifact_reason("Δ Θ Σ α β γ δ") == "non_persian_script"


# ── VLM artifact detection: legitimate labels must NOT be rejected ──────────


@pytest.mark.parametrize(
    "raw",
    [
        "سلام دنیا",
        "IRA N. LEVINE",                       # REAL: printed English in the scans
        "(کارشناس عمران دانشگاه صنعتی امیرکبیر)",  # REAL: aligned=True crop
        "۵۰/۰۰۰/۰۰۰",                          # REAL: bare figure on an accounting page
        "تاریخ ۱ آبر ۱ ۹۷",                    # REAL: c24b457fef08.jpg
        "θ = ۳۰",                              # sparse Greek in a math line
        "الف) ۲۵٪",
        f"می{ZWNJ}شود",
    ],
)
def test_legitimate_labels_are_not_artifacts(raw):
    assert not is_vlm_artifact(raw), vlm_artifact_reason(raw)


def test_single_pipe_is_not_a_table():
    assert not is_vlm_artifact("الف | ب")


# ── persian_char_ratio ──────────────────────────────────────────────────────


def test_persian_char_ratio():
    assert persian_char_ratio("سلام") == 1.0
    assert persian_char_ratio("hello") == 0.0
    assert persian_char_ratio("") == 0.0
    assert 0.4 < persian_char_ratio("سلام hello") < 0.6
