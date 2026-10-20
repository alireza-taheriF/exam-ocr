"""Text-only OCR post-correction with a LOCAL LLM (Ollama native chat API), plus code-level guards.

    CORRECTOR_MODEL=gemma3:12b python -m htr_eval.correct --pred htr_eval/cache/qwen3vl-fa.jsonl --subset-pages 20
    python -m htr_eval.harness --pred htr_eval/cache/qwen3vl-fa.gemma3-12b.corrected.jsonl --subset-pages 20

Only OCR text is sent (never images, never gold labels). Every LLM reply is cached on disk.
The LLM may only change words; each change must pass the guards below or the original is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htr_eval.harness import load_jsonl, load_truth, subset_pages  # noqa: E402
from pipeline.normalizer import ZWNJ, normalize_text  # noqa: E402
from training.eval.metrics import levenshtein  # noqa: E402

CACHE = ROOT / "htr_eval" / "cache"
FREQ_LIST = ROOT / "htr_eval" / "data" / "fa_50k.txt"  # hermitdave/FrequencyWords fa 2018, downloaded once
BATCH, CONTEXT = 10, 2
MAX_EDIT_RATIO = 0.5

SYSTEM = (
    "You post-correct OCR output of Persian handwriting. You get JSON with OCR lines and may fix single words.\n"
    "Rules:\n"
    "- Fix a word only when the context clearly demands it: OCR letter confusions (ق/ف/غ, ج/چ/ح/خ, ب/پ/ت/ث/ن/ی), "
    "wrong or missing dots, and misspellings of a real Persian word.\n"
    "- Never add, remove, translate or paraphrase content. Keep the number of words the same.\n"
    "- Keep all numbers and symbols exactly as they are. Never write Latin letters or digits that are not in the input line.\n"
    "- If unsure, leave the word unchanged. Unreadable fragments stay as they are.\n"
    "- context_before / context_after are only for understanding; do not return them.\n"
    'Return only JSON: {"lines":[{"id":<id>,"text":"<corrected line>"}]} with exactly one entry per input line.'
)

# Invented OCR-style errors (not taken from the gold set).
SHOTS = [
    ([{"id": 0, "text": "من مصلن به مدرسه رفتم"}, {"id": 1, "text": "فیمت این کالا بسیار بالا است"}],
     ["من مثلاً به مدرسه رفتم", "قیمت این کالا بسیار بالا است"]),
    ([{"id": 0, "text": "حانه ما نزدیک پارک اسه"}, {"id": 1, "text": "ی ر د ۱۲ ص"}],
     ["خانه ما نزدیک پارک است", "ی ر د ۱۲ ص"]),
    ([{"id": 0, "text": "۱۲۰ نفر در کلاس بودند"}, {"id": 1, "text": "معلم سوال جدید را توضبح داد"}],
     ["۱۲۰ نفر در کلاس بودند", "معلم سوال جدید را توضیح داد"]),
]

_PUNCT = re.compile(r"[^\w]|_", re.UNICODE)
_DIGITS = re.compile(r"[0-9۰-۹٠-٩]")
_LATIN = re.compile(r"[A-Za-z]")


def key(word: str) -> str:
    """Lexicon key: drop punctuation, ZWNJ and tanwin so «مثلاً» == «مثلا»."""
    return _PUNCT.sub("", normalize_text(word).replace(ZWNJ, "").replace("ً", ""))


def build_lexicon() -> set[str]:
    """Train-set words (minus flagged labels and gold pages) + a public frequency list. Never gold text."""
    d = ROOT / "dataset"
    bad = {r["image_path"] for f in ("label_errors", "labels_rejected") for r in load_jsonl(d / f"{f}.jsonl")}
    gold_pages = {r["source_image"] for r in load_jsonl(d / "gold.jsonl")}
    lex: set[str] = set()
    for r in load_jsonl(d / "train.jsonl"):
        if r["source_image"] in bad or r["source_image"] in gold_pages or r["image_path"] in bad:
            continue
        lex.update(key(w) for w in (r.get("text") or "").split())
    lex.update(key(w.split()[0]) for w in FREQ_LIST.read_text(encoding="utf-8").splitlines() if w.strip())
    lex.discard("")
    return lex


def apply_guards(orig: str, new: str, lex: set[str], rej: Counter) -> str:
    """Return `orig` with only the safe word changes of `new` applied; count rejections per guard."""
    o, n = orig.split(), new.split()
    if new.strip() == orig.strip():
        return orig
    if abs(len(n) - len(o)) > 1:
        rej["word_count"] += 1
        return orig
    if (_DIGITS.search(new) and set(_DIGITS.findall(new)) - set(_DIGITS.findall(orig))) or \
            (_LATIN.search(new) and set(_LATIN.findall(new)) - set(_LATIN.findall(orig))):
        rej["new_digit_or_latin"] += 1
        return orig
    out: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if op == "equal":
            out += o[i1:i2]
            continue
        if i1 == i2 or j1 == j2:  # pure insertion or deletion
            rej["word_count"] += 1
            out += o[i1:i2]
        elif levenshtein(ka := " ".join(map(key, o[i1:i2])), kb := " ".join(map(key, n[j1:j2]))) > MAX_EDIT_RATIO * max(len(ka), len(kb)):
            rej["edit_distance"] += 1
            out += o[i1:i2]
        elif any(key(w) not in lex for w in n[j1:j2] if key(w) and not _DIGITS.search(w)):
            rej["vocabulary"] += 1
            out += o[i1:i2]
        else:
            out += n[j1:j2]
    return " ".join(out)


def chat(base: str, model: str, messages: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "format": "json", "options": {"temperature": 0}}
    if model.startswith("qwen3"):
        body["think"] = False
    r = requests.post(f"{base}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]


def _payload(lines: list[dict], before: list[str], after: list[str]) -> str:
    return json.dumps({"context_before": before, "lines": lines, "context_after": after}, ensure_ascii=False)


def correct_batch(base: str, model: str, rows: list[dict], before: list[str], after: list[str], cache: dict, cf) -> list[str] | None:
    """Raw LLM texts for the rows (None if the reply is unusable). Cached by request hash."""
    lines = [{"id": i, "text": r["text"], **({"alternatives": r["candidates"]} if r.get("candidates") else {})}
             for i, r in enumerate(rows)]
    messages = [{"role": "system", "content": SYSTEM}]
    for ins, outs in SHOTS:
        messages += [{"role": "user", "content": _payload(ins, [], [])},
                     {"role": "assistant", "content": json.dumps({"lines": [{"id": i, "text": t} for i, t in enumerate(outs)]}, ensure_ascii=False)}]
    messages.append({"role": "user", "content": _payload(lines, before, after)})
    k = hashlib.sha1(json.dumps([model, messages], ensure_ascii=False).encode()).hexdigest()
    if k not in cache:
        s = time.perf_counter()
        msg = chat(base, model, messages)
        if "<think>" in msg["content"] or msg.get("thinking"):
            raise RuntimeError("thinking text in reply; thinking must be off")
        cache[k] = {"content": msg["content"], "seconds": round(time.perf_counter() - s, 3), "n": len(rows)}
        cf.write(json.dumps({"k": k, **cache[k]}, ensure_ascii=False) + "\n")
        cf.flush()
    try:
        got = {int(x["id"]): str(x["text"]) for x in json.loads(cache[k]["content"])["lines"]}
        return [got[i] for i in range(len(rows))]
    except (ValueError, KeyError, TypeError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--subset-pages", type=int, default=0)
    args = p.parse_args()

    base = os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("CORRECTOR_MODEL", "gemma3:12b")
    if urlparse(base).hostname not in {"localhost", "127.0.0.1"}:
        raise SystemExit(f"refusing non-local CORRECTOR_BASE_URL {base!r}: nothing may leave this machine")

    rows = load_jsonl(args.pred)
    if args.subset_pages:
        keep = {r["image_path"] for r in subset_pages(load_truth(ROOT / "dataset" / "gold.jsonl"), args.subset_pages)}
        rows = [r for r in rows if r["image_path"] in keep]
    lex = build_lexicon()
    tag = f"{args.pred.stem}.{model.replace(':', '-')}"
    cache_path = CACHE / f"{tag}.llm.jsonl"
    cache = {r["k"]: r for r in load_jsonl(cache_path)} if cache_path.exists() else {}
    rej: Counter = Counter()
    stats = Counter()
    out_rows = []
    with cache_path.open("a", encoding="utf-8") as cf:
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            before = [r["text"] for r in rows[max(0, i - CONTEXT):i]]
            after = [r["text"] for r in rows[i + BATCH:i + BATCH + CONTEXT]]
            raw = correct_batch(base, model, chunk, before, after, cache, cf)
            if raw is None:
                rej["bad_json"] += len(chunk)
                raw = [r["text"] for r in chunk]
            for r, new in zip(chunk, raw):
                text = apply_guards(r["text"], new, lex, rej)
                stats["llm_changed"] += new.strip() != r["text"].strip()
                stats["applied"] += text != r["text"]
                out_rows.append({**r, "text": text, "text_raw": r["text"]})
    out = CACHE / f"{tag}.corrected.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    secs = sum(cache[k]["seconds"] for k in cache if cache[k].get("n")) / max(1, sum(cache[k]["n"] for k in cache))
    summary = {"lines": len(rows), **stats, "rejected": dict(rej), "seconds_per_line": round(secs, 3)}
    out.with_suffix(".stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{out.name}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
