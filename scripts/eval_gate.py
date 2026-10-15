#!/usr/bin/env python3
"""Evaluation gate: score one engine configuration on the gold set and compare
it with the champion (see .claude/skills/ocr-eval-gate).

Runs inference through pipeline/engines.py, so what is measured is exactly what
the app serves. Predictions are flushed per crop; an interrupted run resumes
(only if the configuration is unchanged).

    HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/eval_gate.py --label base_v1
    ... --label rep105 --repetition-penalty 1.05
    ... --label lora_v2 --adapter models/lora_v2 --register

Writes dataset/reports/<label>.json (+ <label>.preds.jsonl). With --register,
the result is recorded in models/registry.json as `champion` (if it passes the
promotion rule against the current champion) or `rejected`. The first
registered result becomes the champion.

All 240 gold rows are inferred: the 168 legible ones are scored (CER/WER), and
the 72 illegible ones (empty label) measure routing — whether the engine
returns "" for crops a human could not read either.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.engines import QwenSettings  # noqa: E402
from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import levenshtein  # noqa: E402

DEFAULT_GOLD = PROJECT_ROOT / "dataset" / "gold.jsonl"
REPORTS = PROJECT_ROOT / "dataset" / "reports"
REGISTRY = PROJECT_ROOT / "models" / "registry.json"

# Gold documents are whole notebooks; the script type was checked by eye on the
# pages (2026-10-10). All four gold notebooks are handwritten.
SCRIPT_BY_DOCUMENT = {
    "جزوه_دست_نویس_اصول_حسابداری_1": "handwritten",
    "جزوه_ریاضی_پیش__نیکوکار_mathfree": "handwritten",
    "CamScanner_۲۰۲۵-۰۶-۰۳_۲۲_۳۲": "handwritten",
    "مقدمات_راهنمایی_و_مشاوره": "handwritten",
}

BOOTSTRAP_ITERS = 2000
NOISE_POINTS = 1.0  # skill rule: with <500 lines, <1 accuracy point is noise


# ── scoring ─────────────────────────────────────────────────────────────────
def line_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(char_dist, char_len, word_dist, word_len), clamped per line like corpus_cer."""
    r, h = normalize_text(ref or ""), normalize_text(hyp or "")
    rw, hw = r.split(), h.split()
    return (min(levenshtein(r, h), len(r)), len(r),
            min(levenshtein(rw, hw), len(rw)), len(rw))


def rate(rows: list[dict], dist: str = "cd", length: str = "cl") -> float | None:
    total = sum(r[length] for r in rows)
    return round(sum(r[dist] for r in rows) / total, 4) if total else None


def summarize(rows: list[dict]) -> dict:
    c = rate(rows)
    return {"lines": len(rows), "cer": c, "wer": rate(rows, "wd", "wl"),
            "accuracy_pct": round(100 * (1 - c), 2) if c is not None else None}


def length_bucket(n: int) -> str:
    return "<=4" if n <= 4 else ("5-20" if n <= 20 else ">20")


def bootstrap_ci(rows: list[dict], seed: int = 0) -> list[float]:
    rnd = random.Random(seed)
    vals = []
    for _ in range(BOOTSTRAP_ITERS):
        sample = [rows[rnd.randrange(len(rows))] for _ in rows]
        vals.append(sum(r["cd"] for r in sample) / max(1, sum(r["cl"] for r in sample)))
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def paired_delta_ci(a: dict[str, dict], b: dict[str, dict], seed: int = 0) -> dict:
    """CER(b) - CER(a) over the lines both scored; negative = b is better."""
    keys = sorted(set(a) & set(b))
    rnd = random.Random(seed)

    def delta(ks: list[str]) -> float:
        la = sum(a[k]["cl"] for k in ks) or 1
        return sum(b[k]["cd"] for k in ks) / la - sum(a[k]["cd"] for k in ks) / la

    vals = sorted(delta([keys[rnd.randrange(len(keys))] for _ in keys]) for _ in range(BOOTSTRAP_ITERS))
    return {"lines": len(keys), "delta_cer": round(delta(keys), 4),
            "ci95": [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]}


class LabelsEngine:
    """Replays a labels JSONL (keyed by crop file name) so a labeler is scored
    exactly like an OCR engine. A crop with no label counts as an empty read."""

    def __init__(self, path: Path):
        self.texts = {Path(r["image_path"]).name: r.get("text") or ""
                      for r in map(json.loads, path.open(encoding="utf-8")) if r}

    def recognize(self, image, key: str) -> tuple[str, None]:
        return self.texts.get(Path(key).name, ""), None


def build_report(gold: list[dict], preds: dict[str, dict]) -> tuple[dict, dict[str, dict]]:
    scored, illegible = [], []
    for g in gold:
        p = preds[g["image_path"]]
        if normalize_text(g.get("text") or ""):
            cd, cl, wd, wl = line_counts(g["text"], p["text"])
            scored.append({"key": g["image_path"], "doc": g.get("source_document"),
                           "cd": cd, "cl": cl, "wd": wd, "wl": wl, "pred": p})
        else:
            illegible.append(p)

    by = lambda f: {k: summarize(v) for k, v in sorted(_group(scored, f).items())}  # noqa: E731
    overall = summarize(scored)
    routed_legible = sum(1 for r in scored if not normalize_text(r["pred"]["text"]))
    routed_illegible = sum(1 for p in illegible if not normalize_text(p["text"]))
    times = [p["seconds"] for p in preds.values() if p.get("seconds") is not None]
    report = {
        **overall,
        "cer_ci95": bootstrap_ci(scored),
        "exact_match_pct": round(100 * sum(r["cd"] == 0 for r in scored) / len(scored), 2),
        "by_document": by(lambda r: r["doc"]),
        "by_length": by(lambda r: length_bucket(r["cl"])),
        "by_script": by(lambda r: SCRIPT_BY_DOCUMENT.get(r["doc"], "unknown")),
        "legible_without_routed": summarize([r for r in scored if normalize_text(r["pred"]["text"])]),
        "routing": {
            "legible_lines_returned_empty": routed_legible,
            "illegible_lines": len(illegible),
            "illegible_lines_returned_empty": routed_illegible,
            "illegible_lines_with_text": len(illegible) - routed_illegible,
        },
        "seconds_per_line_median": round(statistics.median(times), 2) if times else None,
    }
    return report, {r["key"]: r for r in scored}


def _group(rows: list[dict], key) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


# ── champion / registry ─────────────────────────────────────────────────────
def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"champion": None, "history": []}


def compare(champion_label: str, report: dict, scored: dict[str, dict], gold: list[dict]) -> dict:
    # The champion is re-scored against the *current* gold, not the copy of gold
    # stored in its preds file, so a gold correction moves both sides equally.
    champ_preds = {p["image_path"]: p for p in map(json.loads, (REPORTS / f"{champion_label}.preds.jsonl")
                                                    .open(encoding="utf-8"))}
    champ_report, champ_scored = build_report(gold, champ_preds)
    delta = paired_delta_ci(champ_scored, scored)
    doc_deltas = {}
    for doc, cur in report["by_document"].items():
        old = champ_report["by_document"].get(doc)
        if old and old["accuracy_pct"] is not None:
            doc_deltas[doc] = round(cur["accuracy_pct"] - old["accuracy_pct"], 2)
    gain_points = -100 * delta["delta_cer"]
    checks = {
        "lower_cer": delta["delta_cer"] < 0,
        "ci_excludes_zero": delta["ci95"][1] < 0,
        "beyond_noise_1pt": gain_points >= NOISE_POINTS,
        "no_document_regresses_over_5pt": all(d >= -5 for d in doc_deltas.values()),
        "correction_holdout_not_worse": None,  # no student-correction hold-out data yet
    }
    passed = all(v for v in checks.values() if v is not None)
    return {"champion": champion_label, "paired": delta, "accuracy_gain_points": round(gain_points, 2),
            "document_delta_points": doc_deltas, "checks": checks, "passes_promotion_rule": passed}


# ── main ────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Score an engine configuration on gold and compare with the champion")
    p.add_argument("--label", required=True, help="report name: dataset/reports/<label>.json")
    p.add_argument("--engine", default="qwen_local", choices=["qwen_local", "fake", "labels"])
    p.add_argument("--labels", type=Path, default=None,
                   help="labels JSONL to score as if it were an engine (--engine labels), e.g. a labeler's output")
    p.add_argument("--adapter", default="", help="LoRA adapter dir (qwen_local)")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--champion", default=None, help="label to compare with (default: registry champion)")
    p.add_argument("--register", action="store_true", help="record the result in models/registry.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    d = QwenSettings()
    p.add_argument("--prompt", default=d.prompt)
    p.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--px-per-token", type=float, default=d.px_per_token)
    p.add_argument("--repetition-penalty", type=float, default=d.repetition_penalty)
    p.add_argument("--min-height", type=int, default=d.min_height)
    p.add_argument("--clahe", action="store_true", default=d.clahe)
    p.add_argument("--route-min-width", type=int, default=d.route_min_width)
    p.add_argument("--drop-loops", action="store_true", default=d.drop_loops)
    args = p.parse_args()

    settings = QwenSettings(prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                            px_per_token=args.px_per_token, repetition_penalty=args.repetition_penalty,
                            min_height=args.min_height, clahe=args.clahe,
                            route_min_width=args.route_min_width, drop_loops=args.drop_loops)
    if (args.engine == "labels") != (args.labels is not None):
        p.error("--engine labels and --labels go together")
    config = {"engine": args.engine, "adapter": args.adapter or None, "gold": str(args.gold),
              "settings": dataclasses.asdict(settings)}
    if args.labels:
        config = {"engine": "labels", "labels": str(args.labels), "gold": str(args.gold)}
    config_hash = hashlib.sha1(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]

    REPORTS.mkdir(parents=True, exist_ok=True)
    pred_path = REPORTS / f"{args.label}.preds.jsonl"
    report_path = REPORTS / f"{args.label}.json"

    gold = [json.loads(l) for l in args.gold.open(encoding="utf-8") if l.strip()]
    if args.limit:
        gold = gold[: args.limit]

    preds: dict[str, dict] = {}
    if pred_path.exists() and not args.no_resume:
        for line in pred_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("config_hash") != config_hash:
                sys.exit(f"{pred_path} was produced by another configuration; use a new --label or --no-resume")
            preds[r["image_path"]] = r
    todo = [g for g in gold if g["image_path"] not in preds]
    print(f"label={args.label} config={config_hash} gold={len(gold)} to_run={len(todo)}", flush=True)

    if todo:
        from PIL import Image

        if args.engine == "labels":
            engine = LabelsEngine(args.labels)
        elif args.engine == "fake":
            from pipeline.engines import FakeEngine
            engine = FakeEngine()
        else:
            from pipeline.engines import QwenLocalEngine
            engine = QwenLocalEngine(settings=settings, adapter=args.adapter)
        with pred_path.open("w" if args.no_resume or not preds else "a", encoding="utf-8") as sink:
            for i, g in enumerate(todo, 1):
                image = Image.open(PROJECT_ROOT / g["image_path"]).convert("RGB")
                t0 = time.perf_counter()
                text, conf = engine.recognize(image, key=g["image_path"]) if args.engine == "labels" \
                    else engine.recognize(image)
                row = {"image_path": g["image_path"], "source_document": g.get("source_document"),
                       "gold": g.get("text") or "", "text": text, "confidence": conf,
                       "width": image.width, "height": image.height,
                       "seconds": round(time.perf_counter() - t0, 2), "config_hash": config_hash}
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                preds[g["image_path"]] = row
                if i % 20 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {row['seconds']}s", flush=True)

    report, scored = build_report(gold, preds)
    registry = load_registry()
    champion = args.champion or (registry["champion"] or {}).get("label")
    comparison = None
    if champion and champion != args.label and (REPORTS / f"{champion}.json").exists():
        comparison = compare(champion, report, scored, gold)

    result = {"label": args.label, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config, "config_hash": config_hash, **report, "vs_champion": comparison}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  CER {result['cer']}  accuracy {result['accuracy_pct']}%  CI95(CER) {result['cer_ci95']}")
    for k, v in result["by_length"].items():
        print(f"  len {k:>5}: {v['accuracy_pct']}% ({v['lines']} lines)")
    print(f"  routing: {result['routing']}")
    if comparison:
        print(f"  vs {champion}: Δacc {comparison['accuracy_gain_points']:+.2f} pts, "
              f"ΔCER CI95 {comparison['paired']['ci95']}, passes={comparison['passes_promotion_rule']}")
    print(f"  wrote {report_path}")

    if args.register:
        if args.engine == "labels":
            sys.exit("a labeler is not a servable model; --register is refused for --engine labels")
        entry = {"label": args.label, "report": str(report_path.relative_to(PROJECT_ROOT)),
                 "adapter": args.adapter or None, "settings": config["settings"],
                 "cer": result["cer"], "accuracy_pct": result["accuracy_pct"], "date": result["created"]}
        if registry["champion"] is None:
            entry["status"] = "champion"
            registry["champion"] = entry
        elif comparison and comparison["passes_promotion_rule"]:
            entry["status"] = "champion"
            registry["history"].append({**registry["champion"], "status": "retired"})
            registry["champion"] = entry
        else:
            entry["status"] = "rejected"
        registry["history"].append(entry)
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  registry: {entry['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
