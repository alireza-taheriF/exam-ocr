# Persian HTR: engine and corrector shoot-out (2026-10-20)

Gold = `dataset/gold.jsonl` (168 scorable lines). Subset = 20 pages / 48 lines. All runs local, harness numbers only.
Baseline reproduced exactly: full CER 0.3770 / WER 0.7419; subset CER 0.3493 / WER 0.7671 (stored predictions).

## 1. OCR engines (`python -m htr_eval.run --engine X`, scored by `htr_eval.harness`)

| Engine | Set | CER | WER | Word acc | s/line |
|---|---|---|---|---|---|
| **qwen3vl-fa** (Qwen3-VL-2B Persian, MPS fp16) | subset | 0.3333 | **0.7717** | 22.83% | 1.94 |
| qwen3vl-fa | **full** | 0.3707 | **0.7430** | 25.70% | 2.90 |
| kraken-ppocrv6 (medium, CPU) | subset | **0.3254** | 0.7991 | 20.09% | 0.29 |
| surya 0.6.13 recognition (`fa`) | subset | 0.6268 | 0.9589 | 4.11% | 0.37 |
| surya2 (`datalab-to/surya-ocr-2`, base) | subset | 0.5957 | 0.8813 | 11.87% | 1.63 |
| paddleocr-vl | **8 lines only** (3 pages) | 0.3536 | 0.7963 | 20.37% | 43.1 |
| (qwen3vl-fa on the same 8 lines) | 8 lines | 0.2607 | 0.7407 | 25.93% | |

**OCR winner: qwen3vl-fa** (lowest subset WER). The adapter gives CER 0.333 vs the stored 0.349 on the subset (WER 0.772 vs 0.767), so it is equivalent to the baseline. Kraken has the best CER and is 7x faster, but its WER is worse.
PaddleOCR-VL was stopped after 8 lines (30-minute rule: two lines took 129 s and 176 s, WER no better than Qwen); it was NOT run on the 20-page subset.

## 2. Local LLM text correction (Ollama, text only, on qwen3vl-fa subset output)

| Model | CER | WER | Lines improved / worsened | Guard rejections | s/line | WER with guards off |
|---|---|---|---|---|---|---|
| none (OCR only) | 0.3333 | 0.7717 | | | | |
| gemma3:12b | 0.3333 | 0.7717 | 2 / 2 | vocab 5, edit-dist 2, word-count 1 | 2.43 | 0.7671 |
| qwen3:14b | 0.3333 | 0.7763 | 0 / 1 | edit-dist 1, vocab 1, word-count 1, new digit/Latin 1 | 2.63 | 0.7763 |
| qwen3:8b | 0.3333 | 0.7717 | 0 / 0 | none | 1.31 | 0.7717 |

**No local model improves WER on the subset, so the correction step is dropped** (not run on the full set, not added to the pipeline; `OCR_CORRECTION` was not implemented). Even with every guard off the best result is gemma 0.7671 (2 words). The guards are not the bottleneck: the OCR text is too garbled for a 12B model to recover the word (e.g. «خدیات» became «خدای», not «خدمات»). The 0.648 from the hand-run strong-model test is not reproduced.
Edit-distance guard is measured on lexicon keys (tanwin, ZWNJ and punctuation removed), otherwise the target example «مصلن»→«مثلاً» (3/5) would be rejected.

## 3. What helped / what didn't
- Helped: the local pipeline (no data leaves the Mac), per-line OCR with Surya line detection, a decoding-loop filter (cleans output, metrics unchanged: 0.3707 / 0.7430 before and after).
- Didn't: surya 0.6 and surya-ocr-2 base (much worse), PaddleOCR-VL (slow, no better), all three LLM correctors.
- WER 0.85 accuracy (0.15 WER) is far away: best is 0.743 WER on the full set.

## 4. /ponytail-review
No must-fix. Fixed: remote-only script now documents `OCR_ALLOW_REMOTE=1`; added a test for the loop filter. Left: same-row line order is sorted by vertical centre only (unconfirmed risk); the 50k word list is only useful if the corrector is kept. Not checked: README `.venv-ocr` install commands verbatim, long-page load.

## 5. Deviations and caveats
- `.venv-1` is empty here; used `.venv` (OCR), `.venv-surya` (app; extra deps installed with versions pinned), `.venv-kraken` (new).
- `files.pythonhosted.org` was unreachable; packages came from the Tsinghua PyPI mirror. Kraken needed scipy 1.16.3 (1.15.3 would not load on this macOS), so `pip check` warns about kraken's pin.
- Kraken CTC per-word alternatives were not implemented. surya-ocr-2 ran through Hugging Face transformers with surya's own block prompt, not the llama.cpp server; its Urdu-style glyphs (ہ) were not mapped to Persian.
- Lexicon = train.jsonl words (rows from gold pages and flagged labels excluded) + hermitdave fa_50k; duplicate pages could still leak some gold words into it.
- Scripts that call `extract_text_with_vlm` now need `OCR_ALLOW_REMOTE=1`.

## 6. Next experiment
LoRA fine-tune of qwen3vl-fa on real handwriting (`MR3z4/persian-handwriting-ocr`, CC-BY-NC). Caveat from earlier work: LoRA on model-made labels failed and ~55% of real-page crops are fragments, so fix line segmentation/labels before training.
