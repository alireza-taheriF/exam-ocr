"""Persian text normalization for evaluation comparisons.

Applied to both reference and prediction **before** any metric is computed.
Transforms:

* Unify Arabic Yeh/Kaf → Persian ی / ک (also Alef Maqsura → ی)
* Unify ASCII / Arabic-Indic digits → Persian ۰-۹
* Strip tashkeel/diacritics and tatweel
* Strip directional controls; collapse ZWNJ / ZWJ / NBSP variants to nothing
  (or a single space when they sat between letters), then collapse whitespace
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_text", "ZWNJ"]

ZWNJ = "\u200c"

_TATWEEL = "\u0640"
_TASHKEEL = "".join(chr(c) for c in [*range(0x064B, 0x0653), 0x0654, 0x0655, 0x0670])
_FA_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_AR_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_ASCII_DIGITS = "0123456789"

_LETTER_FOLD = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ۀ": "ه",
        "ة": "ه",
        "ۍ": "ی",
        _TATWEEL: None,
    }
)

_DIGIT_FOLD = str.maketrans(
    {**{a: _FA_DIGITS[i] for i, a in enumerate(_ASCII_DIGITS)},
     **{a: _FA_DIGITS[i] for i, a in enumerate(_AR_DIGITS)}}
)

# ZWNJ, ZWJ, LRM/RLM, word joiner, BOM, and other format controls.
_FORMAT_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff\u00ad]")
_TASHKEEL_RE = re.compile(f"[{re.escape(_TASHKEEL)}]")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize Persian text for metric comparison."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _TASHKEEL_RE.sub("", out)
    out = _FORMAT_RE.sub("", out)
    out = out.translate(_LETTER_FOLD)
    out = out.translate(_DIGIT_FOLD)
    out = _SPACE_RE.sub(" ", out).strip()
    return out
