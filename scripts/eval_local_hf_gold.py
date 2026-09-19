#!/usr/bin/env python3
"""Score a local Hugging Face vision model against the hand-verified gold set.

Free, offline, and CPU/MPS-friendly — the point is to find a labeling backend
that costs nothing, now that every paid router is unreachable. Predictions are
appended and flushed per crop, so an interrupted run resumes where it stopped.

    python scripts/eval_local_hf_gold.py \
        --model mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0

The gold set is 168 hand-transcribed lines drawn from the held-out test split,
so a CER measured here is accuracy, not agreement with another auto-labeler.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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

DEFAULT_GOLD = PROJECT_ROOT / "dataset" / "gold.jsonl"
PROMPT = "Transcribe the Persian text in this image. Output only the text."


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="CER of a local HF model on the gold set")
    p.add_argument("--model", required=True)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--pred", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--device", default="mps")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    slug = args.model.split("/")[-1].lower()
    pred_path = args.pred or PROJECT_ROOT / "dataset" / f"pred_{slug}.jsonl"
    report_path = args.report or PROJECT_ROOT / "dataset" / f"report_{slug}.json"

    gold = [r for r in load_jsonl(args.gold) if normalize_text(r.get("text") or "")]
    if args.limit:
        gold = gold[: args.limit]
    print(f"  model : {args.model}")
    print(f"  gold  : {len(gold)} labelled crops")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device
    )
    model.eval()
    print(f"  loaded on {args.device}")

    done: dict[str, dict] = {}
    if pred_path.exists() and not args.no_resume:
        done = {r["image_path"]: r for r in load_jsonl(pred_path)}
    todo = [r for r in gold if r["image_path"] not in done]
    print(f"  to run: {len(todo)}  (already {len(done)})\n")

    mode = "a" if done and not args.no_resume else "w"
    with pred_path.open(mode, encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            crop = PROJECT_ROOT / row["image_path"]
            image = Image.open(crop).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }]
            t0 = time.perf_counter()
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            new = out[0][inputs["input_ids"].shape[1]:]
            text = processor.decode(new, skip_special_tokens=True).strip()
            elapsed = time.perf_counter() - t0

            entry = {
                "image_path": row["image_path"],
                "source_document": row.get("source_document"),
                "text": text,
                "gold": row["text"],
                "cer": round(cer(row["text"], text), 4),
                "seconds": round(elapsed, 2),
            }
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sink.flush()
            done[row["image_path"]] = entry
            print(f"  [{i}/{len(todo)}] {elapsed:5.1f}s  cer={entry['cer']:.2f}  "
                  f"{text[:60]!r}", flush=True)

    refs = [r["text"] for r in gold]
    hyps = [done.get(r["image_path"], {}).get("text") or "" for r in gold]
    times = [done[r["image_path"]]["seconds"] for r in gold if r["image_path"] in done]
    result = {
        "model": args.model,
        "gold": str(args.gold),
        "lines_scored": len(refs),
        "corpus_cer": round(corpus_cer(refs, hyps), 4),
        "corpus_wer": round(corpus_wer(refs, hyps), 4),
        "empty_hypotheses": sum(1 for h in hyps if not normalize_text(h)),
        "seconds_median": round(statistics.median(times), 2) if times else None,
    }
    result["char_accuracy_pct"] = round(accuracy_from_cer(result["corpus_cer"]), 2)
    result["exact_match_pct"] = round(
        100 * sum(cer(r, h) == 0.0 for r, h in zip(refs, hyps)) / len(refs), 2
    )
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\n  corpus CER    : {result['corpus_cer']:.4f}")
    print(f"  corpus WER    : {result['corpus_wer']:.4f}")
    print(f"  char accuracy : {result['char_accuracy_pct']:.2f}%")
    print(f"  exact lines   : {result['exact_match_pct']:.2f}%")
    print(f"  median s/crop : {result['seconds_median']}")
    print(f"\n  wrote {pred_path}\n  wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
