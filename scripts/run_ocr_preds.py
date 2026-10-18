#!/usr/bin/env python3
"""Run the local Qwen3-VL OCR model over a jsonl of crops and write predictions.

Resumable: predictions are appended and flushed per crop, existing rows are
skipped. Only rows with a non-empty `text` are run unless --all is given, so
dev/gold runs never waste time on illegible crops.

    HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/run_ocr_preds.py \
        --data dataset/dev_claude.jsonl --pred dataset/preds/dev_base.jsonl

Score with scripts/eval_ocr.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
PROMPT = "Transcribe the Persian text in this image. Output only the text."


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--adapter", default=None, help="PEFT LoRA directory")
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--dynamic-tokens", type=float, default=0,
                   help="if >0: max_new_tokens = clamp(width/this, 8, 128)")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--max-side", type=int, default=1280,
                   help="downscale crops whose long side exceeds this (memory guard; 0=off)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--all", action="store_true", help="include rows with empty text")
    p.add_argument("--device", default="mps")
    a = p.parse_args()

    rows = load(a.data)
    if not a.all:
        rows = [r for r in rows if (r.get("text") or "").strip()]
    if a.limit:
        rows = rows[: a.limit]
    a.pred.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_path"] for r in load(a.pred)} if a.pred.exists() else set()
    todo = [r for r in rows if r["image_path"] not in done]
    print(f"rows {len(rows)}  todo {len(todo)}", flush=True)
    if not todo:
        return 0

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.float16, device_map=a.device)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    with a.pred.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(todo, 1):
            image = Image.open(ROOT / row["image_path"]).convert("RGB")
            max_new = a.max_new_tokens
            if a.dynamic_tokens > 0:
                max_new = int(min(128, max(8, image.width / a.dynamic_tokens)))
            resized = False
            if a.max_side and max(image.size) > a.max_side:
                k = a.max_side / max(image.size)
                image = image.resize((max(1, int(image.width * k)), max(1, int(image.height * k))))
                resized = True
            t0 = time.perf_counter()
            for attempt in range(3):
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": a.prompt}]}]
                try:
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt").to(model.device)
                    break
                except ValueError:  # very large crop exceeds the processor's token budget
                    f = 0.7
                    image = image.resize((int(image.width * f), int(image.height * f)))
                    resized = True
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    repetition_penalty=a.repetition_penalty)
            text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            sink.write(json.dumps({
                "image_path": row["image_path"], "text": text,
                "seconds": round(time.perf_counter() - t0, 2),
                "max_new_tokens": max_new, "resized": resized}, ensure_ascii=False) + "\n")
            sink.flush()
            if a.device == "mps":
                torch.mps.empty_cache()
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {text[:50]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
