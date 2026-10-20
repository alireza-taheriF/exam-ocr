"""LoRA fine-tune of the Persian Qwen3-VL-2B OCR on hand labels (dataset/human_labels.jsonl).

    .venv/bin/python -m htr_eval.lora_train                 # resumes from the last epoch dir
    LORA_ADAPTER=htr_eval/lora/best .venv/bin/python -m htr_eval.run --engine qwen3vl-lora --subset-pages 20

Loss only on the answer tokens. Validation = ~10% of PAGES (never gold; gold documents are refused).
Each epoch: checkpoint htr_eval/lora/epoch<N>, validation CER; stops when it gets worse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image

from htr_eval.engines import QWEN_FA, HFVisionLM
from htr_eval.harness import ROOT, load_jsonl
from pipeline.normalizer import normalize_text
from scripts.label_web import state
from training.eval.metrics import corpus_cer

OUT = ROOT / "htr_eval" / "lora"


def load_data() -> tuple[list[dict], list[dict]]:
    gold_docs = {r["source_document"] for r in load_jsonl(ROOT / "dataset" / "gold.jsonl")}
    page = {r["image_path"]: r["source_image"] for r in load_jsonl(ROOT / "dataset" / "label_queue.jsonl")}
    rows = [{**r, "page": page[p]} for p, r in state().items() if r["text"].strip() and not r["unreadable"]]
    assert not [r for r in rows if r["source_document"] in gold_docs], "gold document in training data"
    pages = sorted({r["page"] for r in rows}, key=lambda p: hashlib.sha1(p.encode()).hexdigest())
    val_pages = set(pages[: max(1, math.ceil(0.1 * len(pages)))])
    return [r for r in rows if r["page"] not in val_pages], [r for r in rows if r["page"] in val_pages]


def encode(eng: HFVisionLM, img: Image.Image, answer: str) -> dict:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": eng.PROMPT}]}]
    prompt = eng._proc.apply_chat_template(msgs, add_generation_prompt=True)
    full = eng._proc(images=[img], text=[prompt + answer + "<|im_end|>\n"], return_tensors="pt")
    n_prompt = eng._proc(images=[img], text=[prompt], return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100
    return {**full, "labels": labels}


def val_cer(eng: HFVisionLM, val: list[dict]) -> float:
    eng._model.eval()
    hyp = [normalize_text(eng.recognize(Image.open(ROOT / r["image_path"]).convert("RGB"))["text"]) for r in val]
    return corpus_cer([normalize_text(r["text"]) for r in val], hyp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=150)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only N training lines")
    args = ap.parse_args()

    train, val = load_data()
    if args.limit:
        train, val = train[: args.limit], val[: max(2, args.limit // 5)]
    print(f"train {len(train)} lines, val {len(val)} lines ({len({r['page'] for r in val})} pages)")
    random.Random(0).shuffle(train)
    eng = HFVisionLM("lora-train", QWEN_FA, dtype=torch.bfloat16)
    done = sorted(int(p.name[5:]) for p in OUT.glob("epoch*")) if OUT.exists() else []
    if done:
        eng._model = PeftModel.from_pretrained(eng._model, OUT / f"epoch{done[-1]}", is_trainable=True)
    else:
        eng._model = get_peft_model(eng._model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model = eng._model
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    OUT.mkdir(exist_ok=True)
    log = OUT / "log.jsonl"
    best = min((json.loads(l)["val_cer"] for l in log.read_text().splitlines()), default=None) if log.exists() else None
    t_start = time.time()
    prev = best
    for epoch in range(done[-1] + 1 if done else 1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for i, r in enumerate(train, 1):
            batch = encode(eng, Image.open(ROOT / r["image_path"]).convert("RGB"), r["text"]).to(eng._dev)
            loss = model(**batch).loss / args.accum
            loss.backward()
            total, n = total + loss.item() * args.accum, n + 1
            if i % args.accum == 0 or i == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
            if (time.time() - t_start) / 60 > args.max_minutes:
                print("time cap reached")
                return 0
        cer = val_cer(eng, val)
        model.save_pretrained(OUT / f"epoch{epoch}")
        with log.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": round(total / n, 4), "val_cer": round(cer, 4),
                                "minutes": round((time.time() - t_start) / 60, 1)}) + "\n")
        print(f"epoch {epoch}: train loss {total / n:.4f}  val CER {cer:.4f}")
        if prev is None or cer < prev:
            model.save_pretrained(OUT / "best")
        elif cer > prev:
            print("validation CER worsened: stopping early")
            break
        prev = cer if prev is None else min(prev, cer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
