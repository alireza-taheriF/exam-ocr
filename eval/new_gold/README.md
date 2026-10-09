# Sealed new gold set

These line crops are **sealed**: excluded (by path + SHA-256) from every `dataset/qwen_ft/train_mix*.jsonl` manifest.

1. Open each `crop_path` and fill `corrected_text` in `corrections.csv`.
2. Evaluate any model:

```bash
python -m eval.new_gold.eval_sealed --preds path/to/preds.jsonl
```

Predictions JSONL must have `image_path` (or `crop_path`) and `text` (or `pred`).
