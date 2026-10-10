#!/usr/bin/env python3
"""LoRA fine-tune of the Qwen3-VL-2B Persian line-OCR model (built for a free Colab T4).

Base: mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0. Data: real Persian
handwriting (MR3z4/persian-handwriting-ocr) + replay from the base model's own
training set (mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Small) + any
local manifests (student corrections from scripts/export_corrections.py,
labeled val lines). Gold, the test split and eval/new_gold are refused
(training/sealed.py). The prompt is the one the app uses at inference
(pipeline.engines.QwenSettings().prompt), so train and serve match.

    # Colab T4 (see training/lora/README.md)
    python training/lora/train_qwen_lora.py --out /content/drive/MyDrive/exam-ocr/lora_v2 \
        --hf-handwriting 6000 --replay-ratio 0.5 --local /content/data/corrections/train.jsonl

    # Mac dry run: 20 samples, 2 steps, CPU
    python training/lora/train_qwen_lora.py --out /tmp/lora_dry --device cpu \
        --local dataset/val_calib.jsonl --max-samples 20 --max-steps 2 --skip-eval

Outputs in --out: the adapter, processor files, training_meta.json (sources,
counts, data hash, settings, loss history, val CER before/after) and
val_predictions.jsonl. Then score on gold with scripts/eval_gate.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.lora import data as D  # noqa: E402
from training.sealed import sealed_crop_names  # noqa: E402

BASE_MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def prompt_text() -> str:
    from pipeline.engines import QwenSettings

    return QwenSettings().prompt


class LineDataset:
    """Chat-formatted (image, prompt) → text; loss only on the answer tokens."""

    def __init__(self, rows, processor, prompt: str, max_long_side: int, max_label_tokens: int):
        self.rows, self.processor, self.prompt = rows, processor, prompt
        self.max_long_side, self.max_label_tokens = max_long_side, max_label_tokens

    def __len__(self):
        return len(self.rows)

    def _image(self, path):
        from PIL import Image

        image = Image.open(path).convert("RGB")
        if self.max_long_side and max(image.size) > self.max_long_side:
            s = self.max_long_side / max(image.size)
            image = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))), Image.LANCZOS)
        return image

    def __getitem__(self, i):
        row = self.rows[i]
        image = self._image(row["image_path"])
        tok = self.processor.tokenizer
        answer_ids = tok(row["text"], add_special_tokens=False)["input_ids"][: self.max_label_tokens]
        text = tok.decode(answer_ids)
        user = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": self.prompt}]}]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]
        prompt_ids = self.processor.apply_chat_template(
            user, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        enc = self.processor.apply_chat_template(
            full, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        seq_len = enc["input_ids"].shape[1]
        # Per-token tensors (input_ids, attention_mask, mm_token_type_ids, ...) lose
        # the batch dim; vision tensors (pixel_values, image_grid_thw) keep theirs.
        item = {k: v[0] if v.dim() == 2 and v.shape == (1, seq_len) else v for k, v in enc.items()}
        labels = item["input_ids"].clone()
        labels[: prompt_ids["input_ids"].shape[1]] = -100
        item["labels"] = labels
        return item


def collate(features, pad_id: int):
    """Pad text tensors; concatenate Qwen-VL vision tensors (they are flat
    patch lists + one grid row per image, not per-sample batches)."""
    import torch

    max_len = max(f["input_ids"].shape[0] for f in features)

    def pad(t, value):
        return torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=value)

    out = {
        "input_ids": torch.stack([pad(f["input_ids"], pad_id) for f in features]),
        "attention_mask": torch.stack([pad(f["attention_mask"], 0) for f in features]),
        "labels": torch.stack([pad(f["labels"], -100) for f in features]),
    }
    for key in features[0]:
        if key in out:
            continue
        if all(f[key].dim() == 1 and f[key].shape == f["input_ids"].shape for f in features):
            out[key] = torch.stack([pad(f[key], 0) for f in features])  # per-token, e.g. mm_token_type_ids
        else:
            out[key] = torch.cat([f[key] for f in features], dim=0)
    return out


def evaluate(model, processor, rows, prompt: str, max_new_tokens: int = 96) -> tuple[float | None, list[dict]]:
    """Greedy-decode val rows; corpus CER with the project metric."""
    import torch
    from PIL import Image

    from training.eval.metrics import corpus_cer

    if not rows:
        return None, []
    model.eval()
    preds = []
    for r in rows:
        image = Image.open(r["image_path"]).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        inputs = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append({"image_path": r["image_path"], "source": r["source"], "gold": r["text"], "pred": text})
    model.train()
    return round(corpus_cer([p["gold"] for p in preds], [p["pred"] for p in preds]), 4), preds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--device", default="cuda", help="cuda | mps | cpu")
    # data
    p.add_argument("--local", type=Path, nargs="*", default=[], help="local JSONL manifests (image_path, text)")
    p.add_argument("--hf-handwriting", type=int, default=0, help=f"rows from {D.HANDWRITING_DS}")
    p.add_argument("--hf-handwriting-split", default="train")
    p.add_argument("--hw-image-col"); p.add_argument("--hw-text-col"); p.add_argument("--hw-bbox-col")  # noqa: E702
    p.add_argument("--replay-ratio", type=float, default=0.5,
                   help=f"rows from {D.REPLAY_DS} per in-domain/handwriting row")
    p.add_argument("--replay-split", default="train")
    p.add_argument("--rp-image-col"); p.add_argument("--rp-text-col")  # noqa: E702
    p.add_argument("--data-dir", type=Path, default=Path("/content/ocr_data"),
                   help="where streamed HF crops are written")
    p.add_argument("--val-fraction", type=float, default=0.03)
    p.add_argument("--val-max", type=int, default=200)
    p.add_argument("--max-samples", type=int, default=0, help="cap train rows (dry runs)")
    # training
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-long-side", type=int, default=768, help="cap image long side (vision tokens / memory)")
    p.add_argument("--max-label-tokens", type=int, default=128)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-eval", action="store_true", help="no val decoding before/after (dry runs)")
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback, TrainingArguments

    t_start = time.time()
    sealed = sealed_crop_names()
    print(f"[train] sealed crops refused: {len(sealed)}")

    local_rows = [r for m in args.local for r in D.load_local(m, sealed)]
    hw_rows = D.hf_rows(D.HANDWRITING_DS, args.hf_handwriting, args.data_dir / "handwriting",
                        split=args.hf_handwriting_split, seed=args.seed, image_col=args.hw_image_col,
                        text_col=args.hw_text_col, bbox_col=args.hw_bbox_col, source="hf_handwriting")
    n_core = len(local_rows) + len(hw_rows)
    n_replay = int(round(args.replay_ratio * n_core))
    rp_rows = D.hf_rows(D.REPLAY_DS, n_replay, args.data_dir / "replay", split=args.replay_split,
                        seed=args.seed, image_col=args.rp_image_col, text_col=args.rp_text_col,
                        source="hf_replay")
    rows = D.mix(local_rows, hw_rows, rp_rows, replay_ratio=args.replay_ratio, seed=args.seed)
    train_rows, val_rows = D.split_val(rows, args.val_fraction, seed=args.seed)
    val_rows = val_rows[: args.val_max]
    if args.max_samples:
        train_rows = train_rows[: args.max_samples]
    if not train_rows:
        raise SystemExit("no training rows: pass --local and/or --hf-handwriting")
    counts = {s: sum(r["source"] == s for r in train_rows) for s in sorted({r["source"] for r in train_rows})}
    print(f"[train] train={len(train_rows)} {counts} val={len(val_rows)}")

    cuda = args.device.startswith("cuda")
    if cuda:  # T4 has no bf16: fp16 base weights, fp32 LoRA weights + AMP
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float16 if args.device == "mps" else torch.float32
    processor = AutoProcessor.from_pretrained(args.base_model)
    model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype=dtype, device_map=args.device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout, bias="none",
                                             task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    for param in model.parameters():  # GradScaler refuses fp16 trainable params
        if param.requires_grad:
            param.data = param.data.float()
    model.print_trainable_parameters()

    prompt = prompt_text()
    val_cer_before, _ = (None, []) if args.skip_eval else evaluate(model, processor, val_rows, prompt)
    print(f"[train] val CER before: {val_cer_before}")

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = LineDataset(train_rows, processor, prompt, args.max_long_side, args.max_label_tokens)
    losses: list[dict] = []

    class LossLog(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_):
            if logs and "loss" in logs:
                losses.append({"step": state.global_step, "loss": round(float(logs["loss"]), 4)})

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        num_train_epochs=args.epochs, max_steps=args.max_steps, warmup_steps=0.03,  # float = ratio (transformers 5)
        lr_scheduler_type="cosine", logging_steps=1 if args.max_steps and 0 < args.max_steps <= 20 else 10,
        save_steps=args.save_steps, save_total_limit=2, eval_strategy="no",
        remove_unused_columns=False, dataloader_pin_memory=False, report_to=[], seed=args.seed,
        fp16=cuda and dtype == torch.float16, bf16=cuda and dtype == torch.bfloat16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=args.device == "cpu",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda f: collate(f, pad_id), callbacks=[LossLog()])
    resume = any(args.out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))

    val_cer_after, val_preds = (None, []) if args.skip_eval else evaluate(model, processor, val_rows, prompt)
    print(f"[train] val CER after: {val_cer_after}")
    with (args.out / "val_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in val_preds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    finite = [l["loss"] for l in losses if math.isfinite(l["loss"])]
    meta = {
        "base_model": args.base_model, "prompt": prompt,
        "sources": {"local": [str(m) for m in args.local], "hf_handwriting": D.HANDWRITING_DS,
                    "hf_replay": D.REPLAY_DS, "replay_ratio": args.replay_ratio},
        "train_counts": counts, "n_train": len(train_rows), "n_val": len(val_rows),
        "train_data_sha256": D.manifest_hash(train_rows), "val_data_sha256": D.manifest_hash(val_rows),
        "sealed_crops_refused": len(sealed),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "dtype": str(dtype), "steps": trainer.state.global_step,
        "loss_first": finite[0] if finite else None, "loss_last": finite[-1] if finite else None,
        "loss_history": losses, "val_cer_before": val_cer_before, "val_cer_after": val_cer_after,
        "minutes": round((time.time() - t_start) / 60, 1),
        "next": f"scripts/eval_gate.py --label <name> --adapter models/<name> --register",
    }
    (args.out / "training_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[train] saved adapter + training_meta.json -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
