"""Score every engine on the DastKhat 50 with htr_eval.harness' scorer and print ONE table.

    .venv/bin/python bakeoff/score_manual.py
Automated engines: htr_eval/cache/<engine>.dk50.jsonl.  Manual services: bakeoff/manual/<service>/NN.txt
(NN = 01..20, matching bakeoff/manual_20/NN.png; e.g. eboo, matnyaar, google_docs, google_lens).

Every image has a printed template caption «فضای نوشتن» under the handwriting that is not in the truth.
It is removed from EVERY engine's output (fuzzy, so misreads of it go too) before scoring.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from htr_eval.harness import load_jsonl, load_truth, score  # noqa: E402
from training.eval.metrics import levenshtein  # noqa: E402

TRUTH = ROOT / "bakeoff" / "dastkhat_50"
CACHE = ROOT / "htr_eval" / "cache"


def strip_caption(text: str) -> str:
    w, out, i = text.split(), [], 0
    while i < len(w):
        if levenshtein(w[i], "فضای") <= 1 and i + 1 < len(w) and levenshtein(w[i + 1], "نوشتن") <= 2:
            i += 2
        else:
            out.append(w[i])
            i += 1
    return " ".join(out)


def row(name: str, truth: list[dict], preds: dict[str, str], secs: dict[str, float], note: str = "") -> str:
    got = [r for r in truth if r["image_path"] in preds]
    s = score(got, {k: strip_caption(v) for k, v in preds.items()})
    t = [secs[r["image_path"]] for r in got if r["image_path"] in secs]
    sec = f"{sum(t) / len(t):.2f}" if t else "manual"
    flag = f" ({note})" if note else ""
    return f"| {name}{flag} | {len(got)} | {s['cer']:.4f} | {s['wer']:.4f} | {s['word_accuracy_pct']:.1f}% | {sec} |"


def main() -> None:
    truth = load_truth(TRUTH)
    rows = []
    for f in sorted(CACHE.glob("*.dk50.jsonl")):
        recs = load_jsonl(f)
        preds = {r["image_path"]: r.get("text") or "" for r in recs}
        n = len({r["image_path"] for r in truth} & preds.keys())
        rows.append(row(f.name.removesuffix(".dk50.jsonl"), truth, preds, {r["image_path"]: r["seconds"] for r in recs if "seconds" in r},
                        "" if n == len(truth) else f"only {n}/{len(truth)} images"))
    mp = json.loads((ROOT / "bakeoff" / "manual_20" / "map.json").read_text())
    for d in sorted((ROOT / "bakeoff" / "manual").glob("*/")) if (ROOT / "bakeoff" / "manual").exists() else []:
        preds = {mp[p.stem]: p.read_text(encoding="utf-8") for p in d.glob("*.txt") if p.stem in mp}
        if preds:
            rows.append(row(d.name, truth, preds, {}, f"manual, {len(preds)} images" if len(preds) < 20 else "manual"))
    print("| engine | images | CER | WER | word acc | s/image |\n|---|---|---|---|---|---|")
    print("\n".join(sorted(rows, key=lambda r: float(r.split("|")[4]))))


if __name__ == "__main__":
    main()
