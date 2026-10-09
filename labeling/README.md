# Labeling UI (Phase D-LITE)

Local-only hand-correction UI for the sealed `eval/new_gold` set (300 lines).
Binds to **127.0.0.1** only. Does not train.

## Quick start

```bash
source .venv/bin/activate

# 1) Cache model preds (Qwen3 local; Qwen2.5 via LM Studio if running)
python -m labeling.precompute_preds --models qwen3
# wide crops are resized (--max-long-side 1024) to avoid HF image-token truncation
# optional, needs LM Studio on :1234 with qwen25vl:
python -m labeling.precompute_preds --models qwen25

# 2) Launch UI
python -m labeling.ui --queue new_gold
# → http://127.0.0.1:8765/

# 3) After labeling, score both models
python -m eval.new_gold_report
```

## Expected time

At **10–15 s/line**, 300 lines ≈ **50–75 minutes** of focused labeling
(plus ~5% covert repeats → ~315 presentations ≈ **55–80 minutes**).

## Behaviour

| Feature | Default |
|---|---|
| Queue order | Seeded shuffle (`--seed 42`) |
| Blind mode | First **60** unique items (no model text) |
| Prefill | Remaining **240** with `qwen3-vl-2b-persian` |
| Covert repeats | ~**5%** (not shown as repeats) |
| Keyboard | Enter save · Ctrl+I illegible · Ctrl+Z undo · Esc skip |

Each saved row records `blind_mode`, `region_type` (`body`/`form_field`),
`action`, timing, and appends to `labeling/exports/corrections.jsonl`
(work is never locked in SQLite alone).

## Storage

- SQLite: `labeling/labels.db` — `corrections` (append-only) + `model_preds`
- JSONL mirror: `labeling/exports/corrections.jsonl`
- CSV sync for sealed eval: `eval/new_gold/corrections.csv`

## Sealed set

`eval/new_gold/` is hash-checked against every `dataset/qwen_ft/train_mix*.jsonl`
manifest. Do not train on these crops.
