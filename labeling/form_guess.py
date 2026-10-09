"""Heuristics to guess body vs form_field region type."""

from __future__ import annotations

import re

FORM_KEYWORDS = {
    "date",
    "month",
    "year",
    "subject",
    "name",
    "class",
    "score",
    "page",
    "no",
    "number",
    "id",
    "email",
    "tel",
    "phone",
    "address",
    "تاریخ",
    "موضوع",
    "نام",
    "نمره",
    "صفحه",
    "کلاس",
}

_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def guess_region_type(text: str | None, *, image_hint: str | None = None) -> str:
    """Return ``form_field`` or ``body``.

    Rules (any match → form_field):
    - Latin-letter ratio ≥ 0.45 among letters
    - Normalized text (strip punct) is a known form keyword
    - Very short Latin-only labels like ``Date:`` / ``Year:``
    """
    raw = (text or "").strip()
    if not raw and image_hint:
        # Fall back to filename/path hints only if empty
        hint = (image_hint or "").lower()
        if any(k in hint for k in ("form", "header", "stamp")):
            return "form_field"
        return "body"

    letters = [c for c in raw if c.isalpha() or "\u0600" <= c <= "\u06FF"]
    if letters:
        latin = sum(1 for c in letters if _LATIN_RE.fullmatch(c))
        if latin / len(letters) >= 0.45:
            return "form_field"

    compact = re.sub(r"[\s:;\-_.]+", "", raw).lower()
    if compact in FORM_KEYWORDS:
        return "form_field"

    # Short English label with trailing colon
    if re.fullmatch(r"[A-Za-z]{2,16}\s*[:;]?", raw):
        return "form_field"

    arabic = len(_ARABIC_RE.findall(raw))
    if arabic == 0 and _LATIN_RE.search(raw) and len(raw) <= 24:
        return "form_field"

    return "body"
