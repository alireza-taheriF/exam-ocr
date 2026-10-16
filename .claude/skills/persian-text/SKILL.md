---
name: persian-text
description: Persian text and RTL conventions for exam-ocr — normalization source of truth, ZWNJ handling, digits, and RTL UI rules for the editable OCR boxes. Use when writing code that compares, stores, normalizes or displays Persian text.
---

# Persian text conventions

- `pipeline/normalizer.py` is the only normalizer. Import it; never re-implement `normalize_text`.
- Store text **as the student/model wrote it** (raw). Normalize only at comparison/metric time and when building training targets, so normalization stays reversible.
- ZWNJ (U+200C) is meaningful (می‌روم ≠ می روم): never strip it, never map it to space. Text inputs must allow typing it (Shift+Space hint in UI).
- آ, ئ, ؤ are real letters and are preserved. ك→ک, ي→ی fold.
- Digits unify to Persian (۰–۹) for comparison; keep raw digits in storage.
- Transcription rule for labels: what is written, not what should have been written. Students fixing OCR should match the sheet, including the author's own spelling mistakes — say this in the UI help text.

## RTL UI

- Every text input/textarea: `dir="rtl"` and `lang="fa"`; page root `<html lang="fa" dir="rtl">`.
- Use a font with good Persian glyph coverage (Vazirmatn from Google Fonts).
- Mixed content (formulas, Latin terms) relies on the Unicode bidi algorithm; do not insert manual RLM/LRM marks into stored text.
