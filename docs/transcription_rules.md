# Transcription rules (Claude labeler)

General rules only; they were not tuned against individual gold rows.

1. **Write what is on the crop, as written** — the author's spelling, digits and mistakes are kept.
2. **Write whatever characters are recognisable**, even a single letter, digit, `ص`, `۰` or a
   circled number. Leave a crop empty (`illegible`) only when *no* character can be recognised
   (pure pen/ring/shadow/blank, or ink with no letterform).
3. **Small/flat crops:** view at 2–3× enlargement.
4. **Page context** (a few lines above/below in `source_image`) may be used to disambiguate digits,
   technical terms and half-cut words, but **only text inside the crop box is transcribed**.
   Words cut at the crop edge: write the visible part if ≥ ~half of the letterforms is visible.
5. **Spaces and punctuation:** no space before `: . ، ؛`; operators written compactly (`x=1`);
   keep item numbers/markers as written (`۱)`, `-`, `*`). Arrows are not transcribed. These
   differences are also neutralised in the lenient metric (`training/eval/metrics.py::lenient_normalize`).
6. **Digits:** any digit system accepted; the metric unifies them. Prefer Persian digits for Persian text,
   ASCII for Latin/math.
7. **Latin text** (printed English such as `Subject:`, formulas) is transcribed in Latin letters.
8. **Multi-line blocks:** if the crop contains several text lines, transcribe all of them, top to bottom, joined by a single space.
9. **Uncertain word inside a readable line:** write the best reading; do not drop the line.
