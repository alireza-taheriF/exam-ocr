# TrOCR Fine-tuning on Google Colab

## Files

| File | Purpose |
|------|---------|
| `train_trocr_fa.ipynb` | Full Colab notebook — run all cells |
| `metrics.py` | CER helpers (mirror of `training/eval/metrics.py`) |
| `export_for_inference.py` | Local smoke test after download |

## Before Colab

On your Mac (already done if you have the zip):

```bash
cd /Users/alireza/exam-ocr
zip -r trocr_training.zip dataset/train.jsonl dataset/val.jsonl dataset/lines
```

Upload `trocr_training.zip` to **Google Drive → `MyDrive/exam-ocr/`**

## In Colab

1. **File → Upload notebook** → choose `training/colab/train_trocr_fa.ipynb`
   - Or upload to Drive and open from Colab
2. **Runtime → Change runtime type → T4 GPU**
3. Run all cells top to bottom (if OOM: **Runtime → Restart session**, then re-run all cells)

Default paths (edit cell 2 if different):

```
ZIP_PATH = /content/drive/MyDrive/exam-ocr/trocr_training.zip
OUTPUT   = /content/drive/MyDrive/exam-ocr/trocr-fa-exam-v1/
```

## After training

Download `MyDrive/exam-ocr/trocr-fa-exam-v1.zip` and:

```bash
unzip trocr-fa-exam-v1.zip -d models/
```

Add to `.env`:

```env
TROCR_MODEL_PATH=models/trocr-fa-exam-v1
OCR_ENGINE=hybrid
TROCR_CONFIDENCE_THRESHOLD=0.7
```

Test:

```bash
python training/colab/export_for_inference.py models/trocr-fa-exam-v1
python training/eval/evaluate.py --dataset dataset/test.jsonl
```

## Hyperparameters (in notebook cell 3)

| Param | Default |
|-------|---------|
| Base | `microsoft/trocr-base-handwritten` |
| Tokenizer | `xlm-roberta-base` |
| Epochs | 20 (early stopping) |
| Micro-batch | 1 (T4) / 4 (L4) / 8 (A100) |
| Grad accum | 8 (T4) → effective batch 8 |
| Optimizer | Adafactor (saves VRAM vs AdamW) |
| max_target_length | 96 (T4) / 128 (A100) |
| LR | 5e-5 |
| Gradient checkpointing | on |
| CER during train | no — cell 10 after training |

Target: val CER < 0.15 (~85% accuracy) for v1.
