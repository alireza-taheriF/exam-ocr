#!/usr/bin/env python3
"""LoRA fine-tune of Qwen3-VL Persian OCR on line crops (recognition only).

Uses document-split train mix from ``scripts/build_qwen_train_mix.py``.
Tune on gold val only; do not look at gold_test during training.

Usage:
  python training/train_qwen_vl_lora.py \\
      --train dataset/qwen_ft/train_mix.jsonl \\
      --val dataset/qwen_ft/val_gold.jsonl \\
      --output models/qwen3-vl-fa-exam-lora-v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
PROMPT = "Transcribe the Persian text in this image. Output only the text."


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass
class TrainConfig:
    model_id: str = DEFAULT_MODEL
    train_path: Path = PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix.jsonl"
    val_path: Path = PROJECT_ROOT / "dataset" / "qwen_ft" / "val_gold.jsonl"
    output_dir: Path = PROJECT_ROOT / "models" / "qwen3-vl-fa-exam-lora-v1"
    device: str = "mps"
    max_samples: int = 0
    num_train_epochs: float = 1.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_new_tokens: int = 96
    logging_steps: int = 20
    save_steps: int = 200
    seed: int = 42


class LineOcrDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        project_root: Path,
        processor,
        max_length: int = 128,
        max_long_side: int = 0,
    ):
        self.max_long_side = max_long_side
        self.rows = rows
        self.project_root = project_root
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image = Image.open(self.project_root / row["image_path"]).convert("RGB")
        if self.max_long_side > 0 and max(image.size) > self.max_long_side:
            scale = self.max_long_side / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        text = (row.get("text") or "").strip()
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        full_messages = user_messages + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
        ]
        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Squeeze batch dim from processor; collator will re-pad.
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = item["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        labels[:prompt_len] = -100
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        item["labels"] = labels
        return item


def _collate(features: list[dict], processor) -> dict:
    # Pad variable-length sequences.
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_len = max(f["input_ids"].shape[0] for f in features)
    batch: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    pixel_keys = [k for k in features[0] if k not in batch]

    for f in features:
        ids = f["input_ids"]
        mask = f.get("attention_mask", torch.ones_like(ids))
        labels = f["labels"]
        pad_n = max_len - ids.shape[0]
        if pad_n > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_n), value=pad_id)
            mask = torch.nn.functional.pad(mask, (0, pad_n), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_n), value=-100)
        batch["input_ids"].append(ids)
        batch["attention_mask"].append(mask)
        batch["labels"].append(labels)

    out = {k: torch.stack(v) for k, v in batch.items()}
    # Vision tensors: stack when shapes match; otherwise keep list (rare).
    for key in pixel_keys:
        vals = [f[key] for f in features]
        try:
            out[key] = torch.stack(vals)
        except Exception:  # noqa: BLE001
            out[key] = vals
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", type=Path, default=TrainConfig.train_path)
    p.add_argument("--val", type=Path, default=TrainConfig.val_path)
    p.add_argument("--output", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--device", default=os.environ.get("QWEN_OCR_DEVICE", "mps"))
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-long-side",
        type=int,
        default=0,
        help="Downscale crops whose long side exceeds this (px); caps image tokens / OOM. 0 = off.",
    )
    args = p.parse_args(argv)

    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
    )

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val) if args.val.exists() else []
    if args.max_samples > 0:
        train_rows = train_rows[: args.max_samples]
    if not train_rows:
        raise SystemExit(f"No training rows in {args.train}")

    print(f"[train] model={args.model}")
    print(f"[train] train_rows={len(train_rows)} val_rows={len(val_rows)}")
    print(f"[train] output={args.output}")

    processor = AutoProcessor.from_pretrained(args.model)
    # Prefer float16 on MPS; bfloat16 when CUDA supports it.
    dtype = torch.float16
    if args.device.startswith("cuda") and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=args.device,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = LineOcrDataset(train_rows, PROJECT_ROOT, processor, max_long_side=args.max_long_side)
    val_ds = LineOcrDataset(val_rows, PROJECT_ROOT, processor, max_long_side=args.max_long_side) if val_rows else None

    args.output.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to=[],
        seed=args.seed,
        fp16=(dtype == torch.float16 and not args.device.startswith("cpu")),
        bf16=(dtype == torch.bfloat16),
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda feats: _collate(feats, processor),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    processor.save_pretrained(str(args.output))

    meta = {
        "base_model": args.model,
        "train": str(args.train),
        "val": str(args.val),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "device": args.device,
    }
    (args.output / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[train] saved adapter → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
