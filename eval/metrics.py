"""Sentence-level similarity and secondary OCR metrics.

Primary metric (after ``eval.normalize.normalize_text``):

    sentence_similarity = clip(1 - levenshtein(pred, ref) / max(len(ref), 1), 0, 1)

Units are Persian sentence spans (``. ؟ ! ؛ ،`` and newlines). When neither side
has reliable sentence punctuation, callers should pass ``unit="line"`` and the
evaluation report must state the line-level fallback.

Alignment uses Needleman–Wunsch over units so inserts/deletes are penalized
(similarity 0) and cannot silently shift subsequent sentences.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

from eval.normalize import normalize_text

__all__ = [
    "sentence_similarity",
    "split_units",
    "align_units",
    "levenshtein",
    "cer",
    "wer",
    "word_accuracy",
    "evaluate_pair",
    "aggregate_results",
    "bootstrap_mean_ci",
    "write_error_csv",
    "AlignedUnit",
    "PairResult",
    "EvalReport",
]

UnitMode = Literal["sentence", "line", "auto"]


def levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
            )
        prev = curr
    return prev[-1]


def sentence_similarity(pred: str, ref: str) -> float:
    """Partial-credit similarity in [0, 1] after normalization."""
    p = normalize_text(pred or "")
    r = normalize_text(ref or "")
    if not r and not p:
        return 1.0
    dist = levenshtein(p, r)
    sim = 1.0 - dist / max(len(r), 1)
    return max(0.0, min(1.0, sim))


def split_units(text: str, *, mode: UnitMode = "sentence") -> list[str]:
    """Split text into sentence or line units (raw, not normalized)."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    if mode == "line":
        return [ln.strip() for ln in raw.split("\n") if ln.strip()]
    return _split_sentences(raw)


def _split_sentences(text: str) -> list[str]:
    """Split on Persian/ASCII sentence punctuation and newlines; keep non-empty."""
    units: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in ".؟!؛،\n":
            unit = "".join(buf).strip()
            # Drop units that are only a delimiter.
            if unit and not all(c in ".؟!؛،" for c in unit):
                units.append(unit)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        units.append(tail)
    return units


def has_sentence_boundaries(text: str) -> bool:
    return bool(re.search(r"[.؟!؛،]", text or ""))


def choose_unit_mode(ref: str, pred: str) -> UnitMode:
    if has_sentence_boundaries(ref) or has_sentence_boundaries(pred):
        return "sentence"
    return "line"


@dataclass
class AlignedUnit:
    page: str
    ref: str
    pred: str
    similarity: float
    region: str = ""
    confidence: float | None = None
    align_tag: str = "match"  # match | delete | insert


def align_units(
    refs: Sequence[str],
    preds: Sequence[str],
    *,
    page: str = "",
    region: str = "",
) -> list[AlignedUnit]:
    """Needleman–Wunsch alignment maximizing sum of sentence_similarity.

    Gaps (missing / extra units) get similarity 0 and are kept in the output.
    """
    n, m = len(refs), len(preds)
    # dp[i][j] = best score aligning refs[:i] with preds[:j]
    neg = -1e9
    dp = [[neg] * (m + 1) for _ in range(n + 1)]
    bt: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0]  # gap score 0
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1]
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + sentence_similarity(preds[j - 1], refs[i - 1])
            delete = dp[i - 1][j]  # missing prediction
            insert = dp[i][j - 1]  # extra prediction
            best = match
            tag = "M"
            if delete > best:
                best, tag = delete, "D"
            if insert > best:
                best, tag = insert, "I"
            dp[i][j] = best
            bt[i][j] = tag

    aligned: list[AlignedUnit] = []
    i, j = n, m
    while i > 0 or j > 0:
        tag = bt[i][j] if i >= 0 and j >= 0 else ("D" if i > 0 else "I")
        if tag == "M":
            ref_u, pred_u = refs[i - 1], preds[j - 1]
            aligned.append(
                AlignedUnit(
                    page=page,
                    ref=ref_u,
                    pred=pred_u,
                    similarity=sentence_similarity(pred_u, ref_u),
                    region=region,
                    align_tag="match",
                )
            )
            i -= 1
            j -= 1
        elif tag == "D":
            aligned.append(
                AlignedUnit(
                    page=page,
                    ref=refs[i - 1],
                    pred="",
                    similarity=0.0,
                    region=region,
                    align_tag="delete",
                )
            )
            i -= 1
        else:
            aligned.append(
                AlignedUnit(
                    page=page,
                    ref="",
                    pred=preds[j - 1],
                    similarity=0.0,
                    region=region,
                    align_tag="insert",
                )
            )
            j -= 1
    aligned.reverse()
    return aligned


def cer(reference: str, hypothesis: str) -> float:
    r = normalize_text(reference or "")
    h = normalize_text(hypothesis or "")
    if not r:
        return 0.0 if not h else 1.0
    return min(1.0, levenshtein(h, r) / len(r))


def wer(reference: str, hypothesis: str) -> float:
    r = normalize_text(reference or "").split()
    h = normalize_text(hypothesis or "").split()
    if not r:
        return 0.0 if not h else 1.0
    return min(1.0, levenshtein(h, r) / len(r))


def word_accuracy(reference: str, hypothesis: str) -> float:
    return max(0.0, min(1.0, 1.0 - wer(reference, hypothesis)))


@dataclass
class PairResult:
    page: str
    ref_text: str
    pred_text: str
    unit_mode: str
    units: list[AlignedUnit] = field(default_factory=list)
    mean_sentence_similarity: float = 0.0
    exact_match_rate: float = 0.0
    high_sim_rate: float = 0.0  # similarity >= 0.80
    cer: float = 0.0
    wer: float = 0.0
    word_accuracy: float = 0.0
    confidence: float | None = None


def evaluate_pair(
    pred: str,
    ref: str,
    *,
    page: str = "",
    region: str = "",
    unit_mode: UnitMode = "auto",
    confidence: float | None = None,
) -> PairResult:
    mode: UnitMode = unit_mode
    if mode == "auto":
        mode = choose_unit_mode(ref, pred)
        # When auto falls back, treat newline-separated page text as line units.
        if mode == "line":
            pass

    if mode == "line":
        # Prefer explicit newlines; if absent, each side is a single unit.
        ref_units = split_units(ref, mode="line") or (
            [ref.strip()] if (ref or "").strip() else []
        )
        pred_units = split_units(pred, mode="line") or (
            [pred.strip()] if (pred or "").strip() else []
        )
        # Gold line crops often arrive as one string per page joined by newlines.
        if len(ref_units) <= 1 and "\n" not in (ref or "") and (ref or "").strip():
            ref_units = [ref.strip()]
        if len(pred_units) <= 1 and "\n" not in (pred or "") and (pred or "").strip():
            pred_units = [pred.strip()]
    else:
        ref_units = split_units(ref, mode="sentence")
        pred_units = split_units(pred, mode="sentence")

    units = align_units(ref_units, pred_units, page=page, region=region)
    for u in units:
        u.confidence = confidence

    sims = [u.similarity for u in units]
    mean_sim = sum(sims) / len(sims) if sims else (1.0 if not ref.strip() and not pred.strip() else 0.0)
    exact = [
        u
        for u in units
        if normalize_text(u.ref) == normalize_text(u.pred) and normalize_text(u.ref)
    ]
    # Exact-match rate over reference-bearing units (+ inserts counted as non-exact).
    denom = len(units) or 1
    exact_rate = len(exact) / denom
    high = sum(1 for s in sims if s >= 0.80) / denom

    return PairResult(
        page=page,
        ref_text=ref,
        pred_text=pred,
        unit_mode=mode,
        units=units,
        mean_sentence_similarity=mean_sim,
        exact_match_rate=exact_rate,
        high_sim_rate=high,
        cer=cer(ref, pred),
        wer=wer(ref, pred),
        word_accuracy=word_accuracy(ref, pred),
        confidence=confidence,
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile bootstrap 95% CI for the mean of ``values``.

    Returns ``mean``, ``ci_low``, ``ci_high``, and ``n``. Empty input → zeros.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}
    mean = sum(vals) / n
    if n == 1:
        return {
            "mean": mean,
            "ci_low": mean,
            "ci_high": mean,
            "n": 1,
            "n_boot": n_boot,
        }
    # Deterministic bootstrap without requiring numpy at import time.
    import random

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += vals[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    lo_i = int((alpha / 2.0) * (n_boot - 1))
    hi_i = int((1.0 - alpha / 2.0) * (n_boot - 1))
    return {
        "mean": mean,
        "ci_low": means[lo_i],
        "ci_high": means[hi_i],
        "n": n,
        "n_boot": n_boot,
    }


@dataclass
class EvalReport:
    primary_metric: str = "mean_sentence_similarity"
    mean_sentence_similarity: float = 0.0
    pct_similarity_ge_0_80: float = 0.0
    exact_match_sentence_rate: float = 0.0
    cer: float = 0.0
    wer: float = 0.0
    word_accuracy: float = 0.0
    n_pages: int = 0
    n_units: int = 0
    unit_mode_used: str = "auto"
    unit_fallback_note: str = ""
    pairs: list[PairResult] = field(default_factory=list)
    all_units: list[AlignedUnit] = field(default_factory=list)
    bootstrap_sentence: dict = field(default_factory=dict)
    bootstrap_page: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "primary_metric": self.primary_metric,
            "mean_sentence_similarity": round(self.mean_sentence_similarity, 6),
            "pct_similarity_ge_0_80": round(self.pct_similarity_ge_0_80, 6),
            "exact_match_sentence_rate": round(self.exact_match_sentence_rate, 6),
            "cer": round(self.cer, 6),
            "wer": round(self.wer, 6),
            "word_accuracy": round(self.word_accuracy, 6),
            "n_pages": self.n_pages,
            "n_units": self.n_units,
            "unit_mode_used": self.unit_mode_used,
            "unit_fallback_note": self.unit_fallback_note,
            "target_mean_sentence_similarity": 0.80,
            "bootstrap_sentence_similarity": self.bootstrap_sentence,
            "bootstrap_page_mean_similarity": self.bootstrap_page,
        }


def aggregate_results(
    pairs: Iterable[PairResult],
    *,
    unit_fallback_note: str = "",
    n_boot: int = 1000,
    boot_seed: int = 42,
) -> EvalReport:
    pairs = list(pairs)
    all_units: list[AlignedUnit] = []
    for p in pairs:
        all_units.extend(p.units)

    if all_units:
        mean_sim = sum(u.similarity for u in all_units) / len(all_units)
        high = sum(1 for u in all_units if u.similarity >= 0.80) / len(all_units)
        exact = sum(
            1
            for u in all_units
            if normalize_text(u.ref) == normalize_text(u.pred)
            and bool(normalize_text(u.ref))
        ) / len(all_units)
    else:
        mean_sim = high = exact = 0.0

    # Corpus CER/WER over full page texts.
    refs = [p.ref_text for p in pairs]
    hyps = [p.pred_text for p in pairs]
    total_c_dist = total_c_len = 0
    total_w_dist = total_w_len = 0
    for r, h in zip(refs, hyps):
        rn, hn = normalize_text(r), normalize_text(h)
        if rn:
            total_c_dist += min(levenshtein(hn, rn), len(rn))
            total_c_len += len(rn)
            rw, hw = rn.split(), hn.split()
            total_w_dist += min(levenshtein(hw, rw), len(rw))
            total_w_len += len(rw)
        elif hn:
            total_c_dist += len(hn)
            total_c_len += len(hn)
            total_w_dist += len(hn.split())
            total_w_len += len(hn.split()) or 1

    corpus_cer = total_c_dist / total_c_len if total_c_len else 0.0
    corpus_wer = total_w_dist / total_w_len if total_w_len else 0.0

    modes = {p.unit_mode for p in pairs}
    mode_used = "line" if modes == {"line"} else (
        "sentence" if modes == {"sentence"} else "mixed"
    )

    unit_sims = [u.similarity for u in all_units]
    page_sims = [p.mean_sentence_similarity for p in pairs]
    boot_sentence = bootstrap_mean_ci(unit_sims, n_boot=n_boot, seed=boot_seed)
    boot_page = bootstrap_mean_ci(page_sims, n_boot=n_boot, seed=boot_seed + 1)

    return EvalReport(
        mean_sentence_similarity=mean_sim,
        pct_similarity_ge_0_80=high,
        exact_match_sentence_rate=exact,
        cer=corpus_cer,
        wer=corpus_wer,
        word_accuracy=max(0.0, min(1.0, 1.0 - corpus_wer)),
        n_pages=len(pairs),
        n_units=len(all_units),
        unit_mode_used=mode_used,
        unit_fallback_note=unit_fallback_note,
        pairs=pairs,
        all_units=all_units,
        bootstrap_sentence={
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in boot_sentence.items()
        },
        bootstrap_page={
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in boot_page.items()
        },
    )


def write_error_csv(
    units: Sequence[AlignedUnit],
    path: Path,
    *,
    extra_rows: Sequence[dict] | None = None,
) -> None:
    """Write per-unit errors sorted worst-first by similarity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(units, key=lambda u: (u.similarity, u.page, u.ref))
    fieldnames = [
        "page",
        "region",
        "ref",
        "pred",
        "similarity",
        "align_tag",
        "confidence",
        "image_path",
        "ground_truth",
        "predicted",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for u in rows:
            w.writerow(
                {
                    "page": u.page,
                    "region": u.region,
                    "ref": u.ref,
                    "pred": u.pred,
                    "similarity": f"{u.similarity:.6f}",
                    "align_tag": u.align_tag,
                    "confidence": "" if u.confidence is None else f"{u.confidence:.4f}",
                    "image_path": u.page,
                    "ground_truth": u.ref,
                    "predicted": u.pred,
                }
            )
        if extra_rows:
            for row in extra_rows:
                w.writerow(row)
