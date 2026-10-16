---
name: feedback-learning
description: Data contract and protocol for turning student edits of OCR output into training data in exam-ocr — storage schema, label-trust filtering, the correction memory (fast loop), and periodic LoRA retraining with promotion (deep loop). Use when touching corrections storage, the edit UI's save path, learning/*, or retraining scripts.
---

# Learning from student corrections

## Core invariants

- A correction is always tied to **one line crop** (`lines.id`), the **prediction it edited** (`predictions.id`, which records `model_version`), and its author/session. A free-text edit that is not mapped to line crops is never used for training.
- Original OCR text and every edit version are kept (append-only). Never overwrite a prediction.
- Only lines from documents the student **confirmed** are eligible for training.
- Students must have consented to training use; documents without consent are excluded and deletable on request.
- Gold, test split and correction hold-out documents never enter training (see `ocr-eval-gate`).

## Label trust (learning/label_trust.py)

Student text is a noisy label. Each correction gets a trust status: `trusted`, `quarantined`, or `rejected`.
Signals (combine, do not rely on one):
- Edit ratio `cer(prediction, correction)`. Very high ratio on a high-confidence prediction suggests rewriting content, not fixing OCR → quarantine.
- Image–text consistency: model log-likelihood of the corrected text given the crop vs. of the prediction. Corrected text far less likely → quarantine.
- Length vs. crop width plausibility; `is_vlm_artifact`; Persian character ratio.
- Unchanged text on a low-confidence line is a weak "confirmed" signal, kept but down-weighted.
- A random 5% of trusted corrections goes to the admin review queue; the measured label error rate is tracked per model version.

## Fast loop: correction memory (learning/correction_memory.py)

- Align prediction and correction at word level (Levenshtein alignment), extract substitution pairs and new vocabulary, scoped globally and per course.
- A pair is applied in post-processing only when seen from ≥3 independent students, its precision on past corrections is ≥0.9, and applying it does not raise val CER.
- Post-processing never changes a word the model produced with high confidence unless the pair is exact-match.

## Deep loop: retraining (learning/retrain.py)

1. Trigger when ≥N new trusted corrections exist (N configurable, start 500) or manually.
2. Build dataset: trusted corrections + replay sample of public/self-training data (≈1:1) to avoid forgetting. Document-level split.
3. Train a LoRA adapter on the current base (Qwen3-VL-2B), record config and data snapshot hash.
4. Run the `ocr-eval-gate` promotion rule. Promote or reject; write both outcomes to `models/registry.json`.
5. Rollback = set the previous version active in the registry; no retraining needed.
