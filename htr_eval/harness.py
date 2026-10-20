"""Score OCR predictions against line-level truth: WER, CER, word accuracy, top word errors.

Truth: a JSONL with ``image_path`` + ``text`` (e.g. dataset/gold.jsonl), or a
directory pair ``<root>/images/*.{jpg,png}`` + ``<root>/truth/<stem>.txt``.
Predictions: a JSONL with ``image_path`` + ``text`` (what htr_eval.run writes).

    python -m htr_eval.harness --pred htr_eval/cache/surya.jsonl --subset-pages 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import corpus_cer, corpus_wer  # noqa: E402

DEFAULT_TRUTH = ROOT / "dataset" / "gold.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_truth(src: Path) -> list[dict]:
    """Return scorable rows: {image_path, text, page}. Empty (illegible) labels are dropped."""
    if src.is_dir():
        rows = []
        for img in sorted((src / "images").iterdir()):
            txt = src / "truth" / f"{img.stem}.txt"
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"} and txt.exists():
                page = img.stem.rsplit("_l", 1)[0]  # <page>_l<NN>.jpg → page
                rows.append({"image_path": str(img.relative_to(ROOT) if img.is_relative_to(ROOT) else img),
                             "text": txt.read_text(encoding="utf-8"), "page": page})
    else:
        rows = [{"image_path": r["image_path"], "text": r.get("text") or "",
                 "page": r.get("source_image") or r["image_path"]} for r in load_jsonl(src)]
    return [r for r in rows if normalize_text(r["text"])]


def subset_pages(rows: list[dict], n_pages: int) -> list[dict]:
    """Deterministic page-level subset (same pages on every run and every engine)."""
    pages = sorted({r["page"] for r in rows}, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    keep = set(pages[:n_pages])
    return [r for r in rows if r["page"] in keep]


def word_errors(refs: list[str], hyps: list[str]) -> Counter:
    errs: Counter = Counter()
    for ref, hyp in zip(refs, hyps):
        r, h = ref.split(), hyp.split()
        for op, i1, i2, j1, j2 in SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
            if op == "equal":
                continue
            errs[(" ".join(r[i1:i2]) or "∅", " ".join(h[j1:j2]) or "∅")] += 1
    return errs


def score(truth: list[dict], preds: dict[str, str]) -> dict:
    refs = [normalize_text(r["text"]) for r in truth]
    hyps = [normalize_text(preds.get(r["image_path"], "")) for r in truth]
    wer = corpus_wer(refs, hyps)
    return {
        "lines": len(truth),
        "pages": len({r["page"] for r in truth}),
        "missing_predictions": sum(r["image_path"] not in preds for r in truth),
        "cer": round(corpus_cer(refs, hyps), 4),
        "wer": round(wer, 4),
        "word_accuracy_pct": round(100 * (1 - wer), 2),
        "top_word_errors": [{"ref": a, "hyp": b, "count": c} for (a, b), c in word_errors(refs, hyps).most_common(20)],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    p.add_argument("--subset-pages", type=int, default=0, help="score only N fixed pages (0 = all)")
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()

    truth = load_truth(args.truth)
    if args.subset_pages:
        truth = subset_pages(truth, args.subset_pages)
    preds = {r["image_path"]: r.get("text") or "" for r in load_jsonl(args.pred)}
    rep = score(truth, preds)

    print(f"{args.pred.name}: pages={rep['pages']} lines={rep['lines']} missing={rep['missing_predictions']}")
    print(f"  CER {rep['cer']:.4f}   WER {rep['wer']:.4f}   word accuracy {rep['word_accuracy_pct']:.2f}%")
    print("  top word errors (ref → hyp ×n):")
    for e in rep["top_word_errors"]:
        print(f"    {e['ref'][:40]} → {e['hyp'][:40]} ×{e['count']}")
    if args.report:
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
