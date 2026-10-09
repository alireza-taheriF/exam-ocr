"""Client/server validation helpers for corrected text."""

from __future__ import annotations

import re

# Persian letters, Arabic presentation forms we fold later, digits, ZWNJ, punct.
DEFAULT_WHITELIST_RE = re.compile(
    r"^["
    r"\u0600-\u06FF"  # Arabic/Persian block
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF"
    r"0-9۰-۹٠-٩"
    r"A-Za-z"  # form fields / Latin tokens allowed
    r"\s"
    r"\u200c\u200d"  # ZWNJ / ZWJ
    r"\.,!?;:،؛؟«»\(\)\[\]\{\}/\-_+=%#@~`'\"…"
    r"]*$"
)


def find_invalid_chars(text: str, pattern: re.Pattern[str] | None = None) -> list[str]:
    pat = pattern or DEFAULT_WHITELIST_RE
    if pat.fullmatch(text or ""):
        return []
    bad: list[str] = []
    # Invert: any char not in the character class
    allowed = pat.pattern
    # Fallback scan
    for ch in text or "":
        if not re.fullmatch(allowed.replace("^[", "[").replace("]*$", "]"), ch):
            # simpler: test single char against a char-class extract
            pass
    # Direct approach
    inner = (
        r"\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
        r"0-9۰-۹٠-٩A-Za-z\s\u200c\u200d"
        r"\.,!?;:،؛؟«»\(\)\[\]\{\}/\-_+=%#@~`'\"…"
    )
    char_ok = re.compile(f"^[{inner}]$")
    for ch in text or "":
        if not char_ok.match(ch):
            bad.append(ch)
    # dedupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for ch in bad:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def validate_correction(
    text: str,
    *,
    action: str,
) -> dict:
    """Return ``{ok, warnings[]}``."""
    warnings: list[str] = []
    stripped = (text or "").strip()
    if action == "marked_illegible":
        return {"ok": True, "warnings": warnings}
    if action == "skipped":
        return {"ok": True, "warnings": warnings}
    if not stripped:
        warnings.append("empty_text_without_illegible")
    bad = find_invalid_chars(text or "")
    if bad:
        warnings.append("non_whitelist_chars:" + "".join(bad[:12]))
    return {"ok": len(warnings) == 0, "warnings": warnings}
