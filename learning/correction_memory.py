"""Fast learning loop: turn repeated student fixes into post-OCR substitutions.

From every trusted (prediction, student text) pair we extract word-level
substitutions ("هسادجاری" → "حسابداری"). A substitution becomes an active rule
only when enough *independent* students made it and it is precise: among all
reviewed lines whose prediction contained the source phrase, the share that the
student changed to the target must reach ``min_precision``. Rules are applied to
new OCR output before the student sees it, and every applied rule is recorded on
the prediction so its effect stays auditable.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable

from learning.label_trust import TRUSTED, trust_for
from pipeline.normalizer import normalize_text

MAX_PHRASE_TOKENS = 3


@dataclass
class Rule:
    src: str
    dst: str
    scope: str                     # "*" (global) or a course name
    count: int
    authors: int
    support: int                   # reviewed lines whose prediction contained src
    precision: float

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class Memory:
    rules: list[Rule] = field(default_factory=list)
    candidates: list[Rule] = field(default_factory=list)

    def rules_for(self, course: str | None) -> list[Rule]:
        return [r for r in self.rules if r.scope == "*" or (course and r.scope == course)]


def tokenize(text: str) -> list[str]:
    return (text or "").split()


def extract_substitutions(predicted: str, final: str) -> list[tuple[str, str]]:
    """Word-aligned (src_phrase, dst_phrase) replacements, each ≤ MAX_PHRASE_TOKENS."""
    a, b = tokenize(predicted), tokenize(final)
    pairs: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "replace":
            continue
        src, dst = a[i1:i2], b[j1:j2]
        if len(src) == len(dst):
            pairs.extend(zip(src, dst))
        elif len(src) <= MAX_PHRASE_TOKENS and len(dst) <= MAX_PHRASE_TOKENS:
            pairs.append((" ".join(src), " ".join(dst)))
    return [(s, d) for s, d in pairs if normalize_text(s) != normalize_text(d)]


def _contains_phrase(tokens: list[str], phrase: list[str]) -> bool:
    n = len(phrase)
    return any(tokens[i : i + n] == phrase for i in range(len(tokens) - n + 1))


def build_memory(
    rows: Iterable[dict[str, Any]],
    *,
    min_authors: int | None = None,
    min_precision: float | None = None,
) -> Memory:
    """Build rules from reviewed lines.

    ``rows`` need ``predicted_text``, ``final_text``, ``confidence``, ``author``
    (or ``doc_author``), ``course``, ``consent``, ``holdout``. Edited and
    confirmed-unedited lines both count: an unedited occurrence of a source
    phrase is evidence *against* the rule.
    """
    if min_authors is None:
        min_authors = int(os.environ.get("MEMORY_MIN_AUTHORS", "3"))
    if min_precision is None:
        min_precision = float(os.environ.get("MEMORY_MIN_PRECISION", "0.9"))

    usable: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("consent") or row.get("holdout"):
            continue
        trust = trust_for(row)
        if trust.status == TRUSTED:
            usable.append(row)

    hits: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "authors": set()})
    for row in usable:
        author = row.get("author") or row.get("doc_author") or f"line:{row.get('line_id')}"
        for src, dst in extract_substitutions(row["predicted_text"], row["final_text"]):
            for scope in ("*", row.get("course")):
                if scope:
                    h = hits[(src, dst, scope)]
                    h["count"] += 1
                    h["authors"].add(author)

    memory = Memory()
    for (src, dst, scope), h in hits.items():
        phrase = src.split()
        support = sum(
            1
            for row in usable
            if (scope == "*" or row.get("course") == scope)
            and _contains_phrase(tokenize(row["predicted_text"]), phrase)
        )
        precision = h["count"] / support if support else 0.0
        rule = Rule(src, dst, scope, h["count"], len(h["authors"]), support, round(precision, 3))
        if rule.authors >= min_authors and precision >= min_precision:
            memory.rules.append(rule)
        else:
            memory.candidates.append(rule)

    # Several targets for the same source: keep only the dominant one per scope.
    best: dict[tuple[str, str], Rule] = {}
    for rule in memory.rules:
        key = (rule.src, rule.scope)
        if key not in best or rule.count > best[key].count:
            best[key] = rule
    memory.rules = sorted(best.values(), key=lambda r: (-len(r.src.split()), -r.count))
    memory.candidates.sort(key=lambda r: -r.count)
    return memory


def apply_rules(text: str, rules: list[Rule]) -> tuple[str, list[dict[str, Any]]]:
    """Apply rules as whole-token matches, longest phrase first, course rules before global."""
    tokens = tokenize(text)
    if not tokens or not rules:
        return text, []
    ordered = sorted(rules, key=lambda r: (-len(r.src.split()), r.scope == "*", -r.count))
    out: list[str] = []
    applied: list[dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        for rule in ordered:
            phrase = rule.src.split()
            if tokens[i : i + len(phrase)] == phrase:
                out.extend(rule.dst.split())
                applied.append({"src": rule.src, "dst": rule.dst, "scope": rule.scope})
                i += len(phrase)
                break
        else:
            out.append(tokens[i])
            i += 1
    return (" ".join(out), applied) if applied else (text, [])
