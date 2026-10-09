# Qwen3-VL LoRA training OOM notes (Phase C)

## Exact config that OOM'd

Run that produced `models/qwen3-vl-fa-exam-lora-v1` (promoted from `checkpoint-200`):

| Setting | Value |
|---|---|
| Script | `training/train_qwen_vl_lora.py` |
| Base model | `mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0` |
| Train manifest | `dataset/qwen_ft/train_mix_indomain.jsonl` (586 rows: 68 gold + 96 gold_oversample + 422 pseudo_qwen) |
| Val | `dataset/qwen_ft/val_gold.jsonl` (60; not used for step selection — `eval_strategy=no`) |
| Device | Apple MPS |
| Precision | `float16` (`fp16=True` in `TrainingArguments`) |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, targets `q/k/v/o` + MLP (`gate/up/down`) |
| Epochs (requested) | 2.0 |
| Optimizer steps (requested) | 294 (`ceil(586/4)` with accum 4, × 2 epochs) |
| Stopped at | step **224 / 294** (~76%), after ~2h16m |
| Error | `RuntimeError: MPS backend out of memory` (MPS allocated ~16.38 GiB; tried +588 MiB; max allowed ~20.13 GiB) |
| Surviving checkpoint | `checkpoint-200` (loss ~0.203 at step 200) → copied to adapter root |
| `per_device_train_batch_size` | **1** |
| `gradient_accumulation_steps` | **4** (effective batch 4) |
| Gradient checkpointing | **enabled** (`model.gradient_checkpointing_enable()` + `TrainingArguments.gradient_checkpointing=True`) |
| Text `max_length` (labels) | **128** (`LineOcrDataset`) |
| Image resolution | **native crop resolution** — no explicit downscale / `max_pixels` cap in the trainer; processor receives full RGB line crops |
| `save_steps` | 200 |
| LR | `1e-4` |

Command:

```bash
python training/train_qwen_vl_lora.py \
  --train dataset/qwen_ft/train_mix_indomain.jsonl \
  --val dataset/qwen_ft/val_gold.jsonl \
  --output models/qwen3-vl-fa-exam-lora-v1 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 4 \
  --lora-r 16 \
  --device mps
```

Root cause pattern: VL attention memory scales with image tokens. A few wide/tall line crops (or dense pages mis-cropped) spike activation memory even with batch 1 + grad checkpointing. MPS watermark then aborts mid-step.

## Three lower-memory configs to try next (not run in Phase C-VERIFY)

### A) Cap visual tokens + keep LoRA fp16 on MPS

- Add a hard image resize in `LineOcrDataset` (e.g. max long side **512** or **768** px) before `processor`.
- Keep `batch_size=1`, raise `grad_accum` to **8** (effective batch 8) so LR schedule stays sane with fewer tokens/step.
- Keep gradient checkpointing on; optionally set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` only if you accept swap risk.
- Expected tradeoff: slightly blurrier digits; much fewer OOM spikes.

### B) QLoRA 4-bit on CUDA (not MPS)

- Move training to a CUDA box (≥12–16 GB).
- Load base with `bitsandbytes` 4-bit (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`).
- LoRA `r=16` (or 8), `batch_size=1`, `grad_accum=8–16`, gradient checkpointing on.
- MPS cannot run bnb 4-bit reliably — this recipe is CUDA-only.
- Expected tradeoff: enables finishing 1–2 epochs on the full mix; needs a GPU host.

### C) Smaller LoRA + shorter targets + 1 epoch

- `lora_r=8`, `lora_alpha=16`, dropout `0.05`.
- `max_length=64` for answer tokens; drop overlong pseudo labels.
- `--epochs 1`, `grad_accum=8`, batch 1, image long-side cap **384–512**.
- Filter train mix to exclude extremely wide crops (`width > 2000`) before training.
- Expected tradeoff: less capacity, but likely completes on 16 GB unified memory without relying on gold_test for early stopping.

## Guidance

Do **not** pick checkpoints using `gold_test`. Use document-split `val` only, then report `gold_test` once. Prefer finishing a lower-memory 1-epoch run over resuming the OOM'd 2-epoch job without an image-token cap.
