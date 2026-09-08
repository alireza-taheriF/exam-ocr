"""Persian text normalization — the single source of truth for this repo.

Every CER/WER computation and every label-ingestion path must import from here.
Five divergent copies of `normalize_text` previously existed (`training/eval/metrics.py`,
`training/colab/metrics.py`, and inlined in all three training scripts); they are
deprecated in favour of this module.

Design decisions that differ from the old implementation, and why:

* **ZWNJ (U+200C) is preserved.** The old normalizer mapped it to a space, which
  erased a real Persian orthographic distinction (می‌روم vs می روم) and meant the
  model was trained on ZWNJ but never scored on it. 11.9% of labels contain one.
* **آ (U+0622) is preserved.** It is a distinct Persian letter (آب), not an Arabic
  hamza variant. Only أ / إ / ٱ fold to ا.
* **ئ (U+0626) and ؤ (U+0624) are preserved** — both are real Persian letters
  (مسئول, مؤسسه).
* **Digits unify to one system** (Persian by default). Three coexisted unnormalized:
  Persian ۰-۹ (26,944 chars), ASCII 0-9 (2,379), Arabic-Indic ٠-٩ (142).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

__all__ = [
    "ZWNJ",
    "normalize_text",
    "is_vlm_artifact",
    "vlm_artifact_reason",
    "persian_char_ratio",
]

DigitTarget = Literal["fa", "ascii", "keep"]

# ── Character tables ────────────────────────────────────────────────────────
ZWNJ = "‌"  # ZERO WIDTH NON-JOINER — load-bearing in Persian, never stripped

_TATWEEL = "ـ"
# Harakat/tashkeel: fathatan..sukun, plus superscript alef and the hamza marks
# that appear as combining diacritics rather than as letters.
_TASHKEEL = "".join(chr(c) for c in [*range(0x064B, 0x0653), 0x0654, 0x0655, 0x0670])

_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"       # U+06F0..U+06F9
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"       # U+0660..U+0669
_ASCII_DIGITS = "0123456789"

# Directional / joining controls that carry no orthographic meaning here.
# U+200C (ZWNJ) is deliberately absent from this set.
_ZAP = "‍‎‏؜‪‫‬‭‮﻿"

_LETTER_FOLD = {
    "ك": "ک",  # ك ARABIC KAF        → ک PERSIAN KEHEH
    "ي": "ی",  # ي ARABIC YEH        → ی PERSIAN YEH
    "ى": "ی",  # ى ALEF MAKSURA      → ی
    "أ": "ا",  # أ HAMZA ABOVE       → ا
    "إ": "ا",  # إ HAMZA BELOW       → ا
    "ٱ": "ا",  # ٱ WASLA             → ا
    "ۀ": "ه",  # ۀ HEH+YEH ABOVE     → ه
    "ۍ": "ی",  # ۍ YEH WITH TAIL     → ی
    "ـ": "",        # ـ TATWEEL           → (removed)
}

# ة is context-dependent (مکة→مکه but قوة→قوت). A lexicon-free normalizer cannot
# decide, so it is a parameter; ه is the Persian convention and the default.
_TEH_MARBUTA = "ة"

_TASHKEEL_RE = re.compile(f"[{_TASHKEEL}]")
_ZAP_RE = re.compile(f"[{_ZAP}]")
# Whitespace to fold into a single ASCII space. The upper bound of the U+2000
# block range is U+200B — stopping short of U+200C so ZWNJ survives.
_SPACE_RE = re.compile("[ \t\n\r\f   -​  　]+")

_PERSIAN_LETTER_RE = re.compile(r"[ؠ-يٮ-ۓۺ-ۿ]")


def _digit_table(target: DigitTarget) -> dict[int, str]:
    if target == "fa":
        dst, srcs = _FA_DIGITS, (_AR_DIGITS, _ASCII_DIGITS)
    elif target == "ascii":
        dst, srcs = _ASCII_DIGITS, (_AR_DIGITS, _FA_DIGITS)
    else:
        return {}
    return {ord(c): dst[i] for src in srcs for i, c in enumerate(src)}


_DIGIT_TABLES: dict[str, dict[int, str]] = {
    t: _digit_table(t) for t in ("fa", "ascii", "keep")
}


def normalize_text(
    text: str,
    *,
    digits: DigitTarget = "fa",
    teh_marbuta: str = "ه",
    strip_tashkeel: bool = True,
) -> str:
    """Normalize Persian text for training-label storage and metric comparison.

    Order matters: NFKC first (folds Arabic presentation forms such as ﻻ → لا and
    the ﷺ ligature), then per-character folding, then whitespace collapse last.

    `digits="keep"` leaves all three digit systems untouched — useful only when
    diagnosing which system a source used.
    """
    if not text:
        return ""

    out = unicodedata.normalize("NFKC", text)
    out = _ZAP_RE.sub("", out)
    if strip_tashkeel:
        out = _TASHKEEL_RE.sub("", out)

    out = out.translate(_DIGIT_TABLES[digits])
    for src, dst in _LETTER_FOLD.items():
        if src in out:
            out = out.replace(src, dst)
    if teh_marbuta != _TEH_MARBUTA:
        out = out.replace(_TEH_MARBUTA, teh_marbuta)

    return _SPACE_RE.sub(" ", out).strip()


# ── VLM transcription-artifact detection ────────────────────────────────────
# Run these on the RAW label, before normalize_text(): NFKC folds ① → "1" and
# ⁴ → "4", so normalization destroys exactly the evidence we are looking for.

_MD_TABLE_RULE_RE = re.compile(r"\|\s*:?-{2,}")
# Enclosed Alphanumerics (①-⓿), dingbat circled digits (❶-➓), and Enclosed CJK
# circled numbers 21-50 (㉑…). Deliberately excludes the plane-1 emoji region.
_CIRCLED_RE = re.compile("[①-⓿❶-➓㉈-㊿]")
# Plane-1 pictographs plus VS-16. Check marks (✓ ✗ ☑), stars, and arrows are NOT
# flagged: those are legitimate grader marks and math notation on exam papers.
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff️]")
_GREEK_CYRILLIC_RE = re.compile("[Ͱ-ϿЀ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_WORDLIKE_RE = re.compile(r"[^\W_]", re.UNICODE)

_GREEK_MAX_RATIO = 0.30


def persian_char_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are Arabic-script letters."""
    dense = [c for c in text if not c.isspace()]
    if not dense:
        return 0.0
    return sum(bool(_PERSIAN_LETTER_RE.match(c)) for c in dense) / len(dense)


def vlm_artifact_reason(text: str) -> str | None:
    """Return a short reason string if `text` looks like a VLM transcription
    artifact rather than a transcription of handwriting, else None.

    Grounded in measured frequencies over the old `train.jsonl` (12,030 rows):
    9.1% of labels contained zero Persian/Arabic characters, 10.3% contained a
    pipe, 0.88% were pure table/rule punctuation, and 12 emoji codepoints
    appeared — none of which can occur in handwritten Persian exam text.
    """
    raw = text or ""
    stripped = raw.strip()
    if not stripped:
        return "empty"
    # Circled glyphs are checked before emoji: ❶-➓ (U+2776-U+2793) fall inside the
    # dingbat block the emoji pattern covers, and "circled" is the precise reason.
    if _CIRCLED_RE.search(stripped):
        return "circled_glyph"
    if _EMOJI_RE.search(stripped):
        return "emoji_or_dingbat"
    if _MD_TABLE_RULE_RE.search(stripped):
        return "markdown_table_rule"
    if stripped.count("|") >= 2:
        return "markdown_table_row"
    if not _WORDLIKE_RE.search(stripped):
        return "punctuation_only"

    dense = [c for c in stripped if not c.isspace()]
    if _GREEK_CYRILLIC_RE.search(stripped):
        ratio = sum(bool(_GREEK_CYRILLIC_RE.match(c)) for c in dense) / len(dense)
        if ratio > _GREEK_MAX_RATIO:
            return "non_persian_script"

    if not _PERSIAN_LETTER_RE.search(stripped):
        # Digit-only and Latin-only lines are both legitimate here: accounting
        # pages carry bare figures (۵۰/۰۰۰/۰۰۰) and the scans contain printed
        # English. Reject only when there is no letter *and* no digit at all.
        has_digit = any(c in _FA_DIGITS or c in _AR_DIGITS or c in _ASCII_DIGITS
                        for c in stripped)
        if not _LATIN_RE.search(stripped) and not has_digit:
            return "no_persian_latin_or_digits"
    return None


def is_vlm_artifact(text: str) -> bool:
    """True if `text` should be rejected at label-ingestion time."""
    return vlm_artifact_reason(text) is not None
