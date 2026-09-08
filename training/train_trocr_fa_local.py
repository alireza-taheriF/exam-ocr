"""
Fine-tune TrOCR for Persian (Farsi) — local version for RTX 4060 Ti (16GB).
Run: python train_trocr_fa_local.py
"""

import gc
import json
import os
import re
import glob
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# ──────────────────────────────────────────────
# 1) EDIT THESE PATHS FOR YOUR LOCAL MACHINE
# ──────────────────────────────────────────────
WORK_DIR = Path("./exam-ocr")                 # root containing dataset/
TRAIN_JSONL = WORK_DIR / "dataset" / "train.jsonl"
VAL_JSONL   = WORK_DIR / "dataset" / "val.jsonl"
LINES_DIR   = WORK_DIR / "dataset" / "lines"

for p in (TRAIN_JSONL, VAL_JSONL, LINES_DIR):
    assert p.exists(), f"Missing: {p} — put your dataset here before running."

print("train:", sum(1 for _ in open(TRAIN_JSONL, encoding="utf-8")))
print("val:  ", sum(1 for _ in open(VAL_JSONL, encoding="utf-8")))
print("line images:", len(list(LINES_DIR.glob("*.jpg"))))


# ──────────────────────────────────────────────
# 2) Configuration
# ──────────────────────────────────────────────
@dataclass
class TrainConfig:
    base_model: str = "microsoft/trocr-base-handwritten"
    tokenizer_name: str = "xlm-roberta-base"
    output_dir: str = "./trocr-fa-exam-v1"
    checkpoint_dir: str = "./trocr-checkpoints"
    max_target_length: int = 96
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8   # effective batch = 1 x 8 = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    early_stopping_patience: int = 3
    fp16: bool = True
    seed: int = 42


cfg = TrainConfig()

if torch.cuda.is_available():
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    gpu_name = torch.cuda.get_device_name(0)

    if gpu_mem_gb >= 40:
        cfg.max_target_length = 128
        cfg.per_device_train_batch_size = 8
        cfg.per_device_eval_batch_size = 4
        cfg.gradient_accumulation_steps = 2
    elif gpu_mem_gb >= 22:
        cfg.max_target_length = 128
        cfg.per_device_train_batch_size = 4
        cfg.per_device_eval_batch_size = 2
        cfg.gradient_accumulation_steps = 2
    elif gpu_mem_gb >= 14:   # RTX 4060 Ti 16GB falls here
        cfg.max_target_length = 128
        cfg.per_device_train_batch_size = 4
        cfg.per_device_eval_batch_size = 2
        cfg.gradient_accumulation_steps = 2
    elif gpu_mem_gb >= 8:
        cfg.max_target_length = 96
        cfg.per_device_train_batch_size = 2
        cfg.per_device_eval_batch_size = 1
        cfg.gradient_accumulation_steps = 4

    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(
        f"Effective batch: {cfg.per_device_train_batch_size} x "
        f"{cfg.gradient_accumulation_steps} = "
        f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}"
    )
else:
    print("Warning: no GPU detected — training will be very slow on CPU")

print(cfg)


def free_gpu_memory() -> None:
    for name in ("trainer", "model", "processor"):
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ──────────────────────────────────────────────
# 3) CER metric (Persian-normalized)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 4) Dataset
# ──────────────────────────────────────────────
def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PersianLineDataset(Dataset):
    def __init__(self, records, root: Path, processor, max_target_length: int):
        self.records = [r for r in records if (root / r["image_path"]).exists()]
        self.root = root
        self.processor = processor
        self.max_target_length = max_target_length
        skipped = len(records) - len(self.records)
        if skipped:
            print(f"Warning: skipped {skipped} rows with missing images")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        image = Image.open(self.root / row["image_path"]).convert("RGB")
        text = str(row.get("text") or "").strip()
        if not text:
            text = " "

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


train_records = load_jsonl(TRAIN_JSONL)
val_records = load_jsonl(VAL_JSONL)
print(f"Loaded {len(train_records)} train, {len(val_records)} val records")


# ──────────────────────────────────────────────
# 5) Load model + Persian tokenizer
# ──────────────────────────────────────────────
free_gpu_memory()

processor = TrOCRProcessor.from_pretrained(cfg.base_model)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
processor.tokenizer = tokenizer

model = VisionEncoderDecoderModel.from_pretrained(cfg.base_model)
model.decoder.resize_token_embeddings(len(tokenizer))

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.decoder.config.vocab_size

# Do NOT set model.config.max_length — breaks generate() in transformers >= 4.46
model.generation_config = GenerationConfig.from_model_config(model.config)
model.generation_config.max_new_tokens = cfg.max_target_length
model.generation_config.num_beams = 1

model.gradient_checkpointing_enable()
model.config.use_cache = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if torch.cuda.is_available():
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after model load: {alloc_gb:.2f} GB")

print(f"Device: {device} | vocab: {len(tokenizer)} | max_len: {cfg.max_target_length}")


# ──────────────────────────────────────────────
# 6) Build datasets
# ──────────────────────────────────────────────
train_ds = PersianLineDataset(train_records, WORK_DIR, processor, cfg.max_target_length)
val_ds = PersianLineDataset(val_records, WORK_DIR, processor, cfg.max_target_length)
print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")


# ──────────────────────────────────────────────
# 7) Train
# ──────────────────────────────────────────────
if "trainer" in globals():
    del trainer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

training_args = Seq2SeqTrainingArguments(
    output_dir=cfg.checkpoint_dir,
    per_device_train_batch_size=cfg.per_device_train_batch_size,
    per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    predict_with_generate=False,
    num_train_epochs=cfg.num_train_epochs,
    learning_rate=cfg.learning_rate,
    warmup_steps=cfg.warmup_steps,
    weight_decay=cfg.weight_decay,
    eval_strategy="steps",
    eval_steps=cfg.eval_steps,
    save_strategy="steps",
    save_steps=cfg.save_steps,
    save_total_limit=2,
    logging_steps=cfg.logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=cfg.fp16 and torch.cuda.is_available(),
    report_to="none",
    seed=cfg.seed,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    label_names=["labels"],
    optim="adafactor",  # much lower VRAM than AdamW on 250k vocab
)


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
)

checkpoints = sorted(glob.glob(f"{cfg.checkpoint_dir}/checkpoint-*"))
resume_ckpt = checkpoints[-1] if checkpoints else None
if resume_ckpt:
    print(f"Resuming from: {resume_ckpt}")
else:
    print("Starting fresh training")

trainer.train(resume_from_checkpoint=resume_ckpt)


# ──────────────────────────────────────────────
# 8) Full validation CER (run after training)
# ──────────────────────────────────────────────
EVAL_BEAMS = 4

if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.eval()
total_cer = 0.0
n = 0
samples_shown = 0

with torch.no_grad():
    for row in val_records:
        img_path = WORK_DIR / row["image_path"]
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        ref = row["text"]
        inputs = processor(image, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=cfg.max_target_length,
            num_beams=EVAL_BEAMS,
            use_cache=False,
        )
        hyp = processor.batch_decode(generated, skip_special_tokens=True)[0]
        total_cer += cer(ref, hyp)
        n += 1
        if samples_shown < 5:
            print("REF:", ref[:80])
            print("HYP:", hyp[:80])
            print("CER:", f"{cer(ref, hyp):.3f}")
            print("-" * 40)
            samples_shown += 1

mean_cer = total_cer / max(n, 1)
print(f"\nFull val CER: {mean_cer:.4f} ({n} samples) | accuracy ~{max(0,(1-mean_cer)*100):.1f}%")


# ──────────────────────────────────────────────
# 9) Save final model locally
# ──────────────────────────────────────────────
OUT_DIR = Path(cfg.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(OUT_DIR)
processor.save_pretrained(OUT_DIR)

meta = {
    "base_model": cfg.base_model,
    "tokenizer": cfg.tokenizer_name,
    "epochs": cfg.num_train_epochs,
    "train_samples": len(train_ds),
    "val_samples": len(val_ds),
    "val_cer": round(mean_cer, 4),
    "date": datetime.utcnow().strftime("%Y-%m-%d"),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
(OUT_DIR / "training_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(f"Saved to {OUT_DIR}")
print(json.dumps(meta, ensure_ascii=False, indent=2))