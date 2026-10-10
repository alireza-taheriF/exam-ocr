---
name: ocr-eval-gate
description: Rules for measuring any OCR change in exam-ocr (engine, prompt, preprocessing, LoRA adapter, post-correction) against the gold set before claiming an accuracy gain or promoting a model. Use whenever a change could affect recognized text or a model version is about to be activated.
---

# OCR evaluation gate

Every claim about accuracy in this repo must come from a real run whose output file is saved under `dataset/reports/`. No estimated number is written as a measured one.

## The sets and what they may be used for

| Set | File | Allowed use |
|-----|------|-------------|
| Gold | `dataset/gold.jsonl` | Final scoring and promotion only. Never training, never threshold tuning, never prompt selection by repeated peeking beyond a handful of candidates. |
| Test split | `dataset/test_v3.jsonl` | Never training. Gold is sampled from it. |
| Val split | `dataset/val_v3.jsonl` | Threshold calibration, prompt/hyper-parameter search, targeted manual labels. |
| Train split | `dataset/train_v3.jsonl` | Training and pseudo-labeling. |
| Correction hold-out | documents with `holdout=1` in the DB | Second promotion criterion. Never training. |

Splits are at **document level**. Any new data (including student corrections) is assigned to a split by document id, not by line.

## How to measure

1. Normalize both sides with `pipeline/normalizer.py::normalize_text` — never a local copy.
2. Score with `training/eval/metrics.py` (`corpus_cer`, `corpus_wer`). Empty gold labels (illegible) are excluded from CER and reported separately as illegible-rate handling.
3. Run `scripts/eval_gate.py --engine <name> [--adapter <path>]` (or `scripts/eval_local_hf_gold.py` until eval_gate exists).
4. Report: overall CER and accuracy (`1-CER`), plus breakdown by document, by gold length bucket (≤4, 5–20, >20 chars), and printed vs handwritten.
5. Change one variable at a time. If two changes are combined, also report each alone.

## Promotion rule (champion / challenger)

A new model version becomes active only if all hold:
- Gold CER is lower than the champion's, and the difference is larger than noise (bootstrap 95% CI over lines does not cross zero; with <500 lines treat <1 point as noise).
- Correction hold-out CER is not worse than the champion's.
- No document in the breakdown regresses by more than 5 points.
- The report JSON and the adapter path are recorded in `models/registry.json`.

Otherwise the version is recorded as `rejected` with its report, and the champion stays.
