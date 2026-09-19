#!/usr/bin/env python3
"""Raw single-pass CER of local Qwen2.5-VL GGUF on the throwaway sample.

Requires LM Studio's server with `qwen2.5-vl-7b-instruct` loaded (the Unsloth
Q4_K_S GGUF + mmproj). Does not build the production OCR pipeline.

    python scripts/eval_qwen_gguf_sample.py
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import normalize_text  # noqa: E402
from training.eval.metrics import (  # noqa: E402
    accuracy_from_cer,
    cer,
    corpus_cer,
    corpus_wer,
)

DEFAULT_GOLD = PROJECT_ROOT / "dataset" / "qwen_test_sample.jsonl"
DEFAULT_PRED = PROJECT_ROOT / "dataset" / "qwen_test_preds.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "dataset" / "qwen_test_report.json"
DEFAULT_BASE = "http://127.0.0.1:1234/v1"
DEFAULT_MODEL = "qwen25vl"
GGUF = (
    Path.home()
    / ".lmstudio/models/unsloth/Qwen2.5-VL-7B-Instruct-GGUF"
    / "Qwen2.5-VL-7B-Instruct-Q4_K_S.gguf"
)
MMPROJ = GGUF.parent / "mmproj-F32.gguf"

PROMPT = (
    "متن فارسی داخل این تصویر را عیناً رونویسی کن. "
    "فقط همان یک خط را برگردان، بدون توضیح."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def transcribe(crop: Path, *, base: str, model: str, timeout: float) -> str:
    b64 = base64.b64encode(crop.read_bytes()).decode()
    mime = "image/png" if crop.suffix.lower() == ".png" else "image/jpeg"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 256,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    return (body["choices"][0]["message"].get("content") or "").strip()


def error_kind(ref: str, hyp: str) -> str:
    nref, nhyp = normalize_text(ref), normalize_text(hyp)
    if not nhyp:
        return "empty"
    if not nref:
        return "skipped-empty-ref"
    if nhyp == nref:
        return "exact"
    if len(nhyp) > len(nref) * 1.5 and nref not in nhyp:
        return "hallucination"
    ref_words, hyp_words = set(nref.split()), set(nhyp.split())
    if ref_words - hyp_words and not (len(nref) - len(nhyp) < 3):
        missing = ref_words - hyp_words
        extra = hyp_words - ref_words
        if missing and not extra:
            return "missing-words"
        if extra and not missing:
            return "inserted-words"
    return "char-confusion"


def main() -> int:
    p = argparse.ArgumentParser(description="CER of local Qwen2.5-VL GGUF")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--pred", type=Path, default=DEFAULT_PRED)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    if not args.gold.exists():
        print(f"Not found: {args.gold}")
        print("Run: python scripts/label_qwen_test_sample.py")
        return 1

    gold_rows = [r for r in load_jsonl(args.gold) if normalize_text(r.get("text") or "")]
    if not gold_rows:
        print(f"{args.gold} has no non-empty labels yet.")
        return 1

    print(f"  gold       : {len(gold_rows)} labelled crops")
    print(f"  gguf       : {GGUF}  exists={GGUF.exists()}")
    print(f"  mmproj     : {MMPROJ}  exists={MMPROJ.exists()}")
    print(f"  endpoint   : {args.base}  model={args.model}")

    done: dict[str, dict] = {}
    if args.pred.exists() and not args.no_resume:
        done = {r["image_path"]: r for r in load_jsonl(args.pred)}

    args.pred.parent.mkdir(parents=True, exist_ok=True)
    remaining = [r for r in gold_rows if r["image_path"] not in done]
    print(f"  to run     : {len(remaining)}  (already {len(done)})")

    with args.pred.open("a" if done and not args.no_resume else "w",
                        encoding="utf-8") as sink:
        for i, row in enumerate(remaining, 1):
            crop = PROJECT_ROOT / row["image_path"]
            print(f"  [{i}/{len(remaining)}] {row['image_path']}", flush=True)
            t0 = time.perf_counter()
            err = ""
            raw = ""
            try:
                raw = transcribe(crop, base=args.base, model=args.model,
                                 timeout=args.timeout)
            except urllib.error.URLError as e:
                err = f"url-error: {e}"
            except TimeoutError as e:
                err = f"timeout: {e}"
            except Exception as e:  # noqa: BLE001 — report, don't crash the run
                err = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            entry = {
                "image_path": row["image_path"],
                "source_document": row.get("source_document"),
                "text": raw,
                "text_normalized": normalize_text(raw),
                "gold": row["text"],
                "seconds": round(elapsed, 3),
                "error": err,
            }
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sink.flush()
            done[row["image_path"]] = entry
            preview = (raw or err or "<empty>")[:80]
            print(f"      {elapsed:6.2f}s  {preview}", flush=True)

    refs, hyps, times, pairs = [], [], [], []
    empty_n = 0
    kinds: dict[str, int] = {}
    for row in gold_rows:
        pred = done.get(row["image_path"], {})
        hyp = pred.get("text") or ""
        refs.append(row["text"])
        hyps.append(hyp)
        if "seconds" in pred:
            times.append(float(pred["seconds"]))
        if not normalize_text(hyp):
            empty_n += 1
        kind = error_kind(row["text"], hyp)
        kinds[kind] = kinds.get(kind, 0) + 1
        pairs.append({
            "image_path": row["image_path"],
            "source_document": row.get("source_document"),
            "gold": normalize_text(row["text"]),
            "pred": normalize_text(hyp),
            "raw_pred": hyp,
            "cer": round(cer(row["text"], hyp), 4),
            "seconds": pred.get("seconds"),
            "kind": kind,
            "error": pred.get("error") or "",
        })

    ccer = corpus_cer(refs, hyps)
    cwer = corpus_wer(refs, hyps)
    report = {
        "model_file": str(GGUF),
        "mmproj_file": str(MMPROJ),
        "runtime": "lmstudio-llama.cpp",
        "endpoint": args.base,
        "model_id": args.model,
        "lines_scored": len(refs),
        "empty_hypotheses": empty_n,
        "corpus_cer": round(ccer, 4),
        "corpus_wer": round(cwer, 4),
        "char_accuracy_pct": round(accuracy_from_cer(ccer), 2),
        "error_kinds": kinds,
        "seconds": {
            "n": len(times),
            "mean": round(statistics.mean(times), 3) if times else None,
            "median": round(statistics.median(times), 3) if times else None,
            "min": round(min(times), 3) if times else None,
            "max": round(max(times), 3) if times else None,
            "per_crop": [round(t, 3) for t in times],
        },
        "examples": sorted(pairs, key=lambda x: -x["cer"])[: args.examples],
        "all_pairs": pairs,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\n  lines scored : {report['lines_scored']}")
    print(f"  empty output : {empty_n}")
    print(f"  corpus CER   : {report['corpus_cer']:.4f}")
    print(f"  corpus WER   : {report['corpus_wer']:.4f}")
    print(f"  char acc     : {report['char_accuracy_pct']:.2f}%")
    print(f"  error kinds  : {kinds}")
    if times:
        print(f"  time s/crop  : mean={report['seconds']['mean']:.2f}  "
              f"median={report['seconds']['median']:.2f}  "
              f"min={report['seconds']['min']:.2f}  max={report['seconds']['max']:.2f}")
        print(f"  per crop     : {report['seconds']['per_crop']}")
    print(f"\n  examples (up to {args.examples}, worst CER first):")
    for ex in report["examples"]:
        print(f"    CER {ex['cer']:.4f}  {ex['kind']}  {ex['seconds']}s")
        print(f"      gold: {ex['gold']}")
        print(f"      pred: {ex['pred']}")
    print(f"\n  wrote {args.pred}")
    print(f"  wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
