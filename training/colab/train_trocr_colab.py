#!/usr/bin/env python3
"""
All-in-one TrOCR training script for Google Colab.

Paste into a single Colab cell (after mounting Drive), or run:
    python train_trocr_colab.py

Requires trocr_training.zip extracted or mounted at WORK_DIR.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit paths if needed
# ═══════════════════════════════════════════════════════════════════════════════

ZIP_PATH = Path("/content/drive/MyDrive/exam-ocr/trocr_training.zip")
WORK_DIR = Path("/content/exam-ocr")
OUTPUT_DIR = Path("/content/drive/MyDrive/exam-ocr/trocr-fa-exam-v1")
CHECKPOINT_DIR = Path("/content/trocr-checkpoints")

BASE_MODEL = "microsoft/trocr-base-handwritten"
TOKENIZER_NAME = "xlm-roberta-base"
MAX_TARGET_LENGTH = 96
NUM_EPOCHS = 20
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 500
LEARNING_RATE = 5e-5
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9").replace("\u200c", " ")
    return re.sub(r"\s+", " ", t).strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PersianLineDataset(Dataset):
    def __init__(self, records: list[dict], root: Path, processor, max_len: int):
        self.records = [r for r in records if (root / r["image_path"]).exists()]
        self.root = root
        self.processor = processor
        self.max_len = max_len
        missing = len(records) - len(self.records)
        if missing:
            print(f"Skipped {missing} rows (missing image)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        text = str(row.get("text") or " ").strip() or " "
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_len, truncation=True
        ).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


class CERCallback(TrainerCallback):
    def __init__(self, records, processor, root, max_samples=50, num_beams=1):
        self.records = records
        self.processor = processor
        self.root = root
        self.max_samples = min(max_samples, len(records))
        self.num_beams = num_beams
        self.best_cer = float("inf")

    def on_evaluate(self, args, state, control, model, **kwargs):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(self.max_samples):
                row = self.records[i]
                img_path = self.root / row["image_path"]
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                inputs = self.processor(image, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs, max_new_tokens=MAX_TARGET_LENGTH, num_beams=self.num_beams
                )
                hyp = self.processor.batch_decode(out, skip_special_tokens=True)[0]
                total += cer(row["text"], hyp)
        mean = total / max(self.max_samples, 1)
        acc = max(0.0, (1.0 - mean) * 100)
        print(f"\n>>> Val CER (n={self.max_samples}): {mean:.4f} | ~{acc:.1f}% acc")
        if mean < self.best_cer:
            self.best_cer = mean
        model.train()


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def ensure_data() -> tuple[Path, Path]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(WORK_DIR)
    train_p = WORK_DIR / "dataset" / "train.jsonl"
    val_p = WORK_DIR / "dataset" / "val.jsonl"
    if not train_p.exists() or not val_p.exists():
        raise FileNotFoundError(f"Missing train/val jsonl under {WORK_DIR}/dataset/")
    return train_p, val_p


def evaluate_full(model, processor, val_records, root) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for row in val_records:
            p = root / row["image_path"]
            if not p.exists():
                continue
            image = Image.open(p).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=MAX_TARGET_LENGTH, num_beams=4)
            hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
            total += cer(row["text"], hyp)
            n += 1
    return total / max(n, 1)


def main() -> None:
    train_p, val_p = ensure_data()
    train_records = load_jsonl(train_p)
    val_records = load_jsonl(val_p)
    print(f"Train: {len(train_records)} | Val: {len(val_records)}")

    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    processor.tokenizer = tokenizer

    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    model.decoder.resize_token_embeddings(len(tokenizer))
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.decoder.config.vocab_size
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.max_new_tokens = MAX_TARGET_LENGTH
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    batch_size = BATCH_SIZE
    grad_accum = GRAD_ACCUM_STEPS
    max_len = MAX_TARGET_LENGTH
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("GPU:", torch.cuda.get_device_name(0), f"({gpu_mem_gb:.1f} GB)")
        if gpu_mem_gb >= 40:
            batch_size, grad_accum, max_len = 8, 2, 128
        elif gpu_mem_gb >= 22:
            batch_size, grad_accum, max_len = 4, 2, 128
        model.cuda()
    print(f"Effective batch: {batch_size} x {grad_accum} = {batch_size * grad_accum}")

    train_ds = PersianLineDataset(train_records, WORK_DIR, processor, MAX_TARGET_LENGTH)
    val_ds = PersianLineDataset(val_records, WORK_DIR, processor, MAX_TARGET_LENGTH)

    args = Seq2SeqTrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        predict_with_generate=False,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        optim="adafactor",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    mean_cer = evaluate_full(model, processor, val_records, WORK_DIR)
    print(f"\nFull val CER: {mean_cer:.4f} | ~{max(0,(1-mean_cer)*100):.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    meta = {
        "base_model": BASE_MODEL,
        "tokenizer": TOKENIZER_NAME,
        "epochs": NUM_EPOCHS,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_cer": round(mean_cer, 4),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "colab_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (OUTPUT_DIR / "training_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved → {OUTPUT_DIR}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
