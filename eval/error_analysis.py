"""Classify OCR errors on gold line pairs for Phase B analysis."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Sequence

from eval.metrics import levenshtein, sentence_similarity
from eval.normalize import normalize_text

# Persian similar-shape confusion pairs (single-char substitutions).
_SIMILAR_PAIRS = {
    frozenset(("ر", "ز")),
    frozenset(("د", "ذ")),
    frozenset(("س", "ش")),
    frozenset(("ص", "ض")),
    frozenset(("ط", "ظ")),
    frozenset(("ع", "غ")),
    frozenset(("ت", "ث")),
    frozenset(("ح", "خ")),
    frozenset(("ج", "چ")),
    frozenset(("ک", "گ")),
    frozenset(("ا", "آ")),
    frozenset(("ه", "ة")),
    frozenset(("ی", "ي")),
    frozenset(("ب", "پ")),
    frozenset(("ف", "ق")),
    frozenset(("ن", "ى")),
}

_DOT_DIACRITIC = set("ًٌٍَُِّْٰٕٖٜۣٓٔٗ٘ٙٚٛٝٞ۟۠ۡۢۤۥۦ۪ۭۧۨ۫۬")
_PUNCT = set(".,?!؛،:؛«»()[]{}|/\\-–—_+*=%#@~`'\"؟٬٫")
_DIGIT = set("۰۱۲۳۴۵۶۷۸۹0123456789٠١٢٣٤٥٦٧٨٩")

CATEGORIES = (
    "empty_output",
    "hallucinated_text",
    "missing_characters",
    "extra_characters",
    "similar_shape_letters",
    "dots_diacritics",
    "joined_letters",
    "numbers_punctuation",
    "other",
)


def _is_arabic_script(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        0x0600 <= o <= 0x06FF
        or 0x0750 <= o <= 0x077F
        or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF
        or 0xFE70 <= o <= 0xFEFF
    )


def _ops(ref: str, pred: str) -> list[tuple[str, str, str]]:
    """Return list of (op, ref_ch, pred_ch) via DP traceback. op in M/S/I/D."""
    r, p = list(ref), list(pred)
    n, m = len(r), len(p)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == p[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] and r[i - 1] == p[j - 1]:
            ops.append(("M", r[i - 1], p[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("S", r[i - 1], p[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("I", "", p[j - 1]))
            j -= 1
        else:
            ops.append(("D", r[i - 1], ""))
            i -= 1
    ops.reverse()
    return ops


def classify_pair(ref: str, pred: str) -> list[str]:
    """Return one or more category labels for a (ref, pred) pair."""
    rn = normalize_text(ref or "")
    pn = normalize_text(pred or "")
    if not rn and not pn:
        return ["other"]
    if not pn:
        return ["empty_output"]
    if not rn:
        return ["hallucinated_text"]

    # Hallucination: pred much longer and low similarity, or mostly non-Persian.
    sim = sentence_similarity(pn, rn)
    arabic_frac = sum(1 for c in pn if _is_arabic_script(c)) / max(len(pn), 1)
    if len(pn) > max(8, 2 * len(rn)) and sim < 0.35:
        return ["hallucinated_text"]
    if arabic_frac < 0.3 and any(_is_arabic_script(c) for c in rn):
        return ["hallucinated_text"]

    ops = _ops(rn, pn)
    labels: list[str] = []
    n_del = sum(1 for o, _, _ in ops if o == "D")
    n_ins = sum(1 for o, _, _ in ops if o == "I")
    n_sub = sum(1 for o, _, _ in ops if o == "S")

    similar = 0
    dots = 0
    num_punct = 0
    for o, a, b in ops:
        if o == "S":
            if frozenset((a, b)) in _SIMILAR_PAIRS:
                similar += 1
            elif a in _DOT_DIACRITIC or b in _DOT_DIACRITIC:
                dots += 1
            elif a in _DIGIT or b in _DIGIT or a in _PUNCT or b in _PUNCT:
                num_punct += 1
            # Dot-related letter pairs often differ by nukta count (ب/ت/ث/پ etc.)
            elif a in "بتثپنی" and b in "بتثپنی" and a != b:
                dots += 1
                similar += 1
        if o in ("I", "D") and ((a in _DIGIT or b in _DIGIT) or (a in _PUNCT or b in _PUNCT)):
            num_punct += 1
        if o in ("I", "D") and (a in _DOT_DIACRITIC or b in _DOT_DIACRITIC):
            dots += 1

    # Joined letters: missing spaces / ZWNJ-like (after normalize ZWNJ gone) —
    # proxy: pred has fewer spaces than ref but similar letter skeleton.
    ref_letters = re.sub(r"\s+", "", rn)
    pred_letters = re.sub(r"\s+", "", pn)
    if (
        rn.count(" ") > pn.count(" ")
        and levenshtein(ref_letters, pred_letters) <= max(2, len(ref_letters) // 5)
        and rn.count(" ") - pn.count(" ") >= 1
    ):
        labels.append("joined_letters")

    if n_del > 0 and n_del >= n_ins:
        labels.append("missing_characters")
    if n_ins > 0 and n_ins > n_del:
        labels.append("extra_characters")
    if similar > 0:
        labels.append("similar_shape_letters")
    if dots > 0:
        labels.append("dots_diacritics")
    if num_punct > 0:
        labels.append("numbers_punctuation")

    # If only substitutions of similar letters, keep those; drop missing/extra noise
    if not labels and (n_sub or n_del or n_ins):
        labels.append("other")
    if not labels and sim < 1.0:
        labels.append("other")
    if not labels:
        labels.append("other")
    return labels


def analyze_units(
    rows: Sequence[dict],
    *,
    top_n: int = 30,
) -> dict:
    """Classify rows with keys ref/pred/similarity/image_path; return summary + worst."""
    counts: Counter[str] = Counter()
    examples: list[dict] = []
    for row in rows:
        ref = row.get("ref") or row.get("ground_truth") or ""
        pred = row.get("pred") or row.get("predicted") or ""
        # Skip empty-ref deletes of punctuation-only if any
        if not normalize_text(ref) and not normalize_text(pred):
            continue
        cats = classify_pair(ref, pred)
        for c in cats:
            counts[c] += 1
        sim = float(row.get("similarity", sentence_similarity(pred, ref)))
        examples.append(
            {
                "image_path": row.get("image_path") or row.get("page") or "",
                "page": row.get("page") or "",
                "region": row.get("region") or "",
                "ref": ref,
                "pred": pred,
                "similarity": sim,
                "categories": "|".join(cats),
            }
        )

    worst = sorted(examples, key=lambda r: (r["similarity"], r["ref"]))[:top_n]
    return {
        "n_examples": len(examples),
        "category_counts": {k: int(counts.get(k, 0)) for k in CATEGORIES},
        "worst": worst,
    }


def write_worst_csv(worst: Sequence[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_path", "page", "region", "ref", "pred", "similarity", "categories"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in worst:
            out = dict(row)
            out["similarity"] = f"{float(out['similarity']):.6f}"
            w.writerow(out)
