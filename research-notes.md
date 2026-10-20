# Persian HTR: research notes

Searched 2026-10-20, time-boxed. Gold = `dataset/gold.jsonl` (168 scorable lines, 4 handwritten lecture-note PDFs, not exam sheets).
"Evidence" means a number from a benchmark or paper; leaderboard or vendor claims are marked weak. Arabic-only evidence is marked **AR**.

| # | Candidate | Link | Date | License | Script / handwriting | Runs locally (M2 Pro 16 GB) | Evidence | Fit |
|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3-VL-2B Persian-Arabic OCR** (current local baseline) | [HF](https://huggingface.co/mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0) | 2025 | check card | FA+AR, printed + some handwriting | yes, 1.45 s/line (MPS) | **Our gold: CER 0.377, WER 0.742**; 20-page subset CER 0.349, WER 0.767 | Baseline. Weak on short lines (CER 0.81 for ≤4 chars) |
| 2 | Qwen2.5-VL-7B (GGUF, LM Studio) | — | 2025 | Apache-2.0 | multilingual | yes, 3.74 s/line | Our gold: CER 0.454, WER 0.772 | Rejected: worse and 2.6× slower than #1 |
| 3 | **Surya OCR 2** (base) | [HF](https://huggingface.co/datalab-to/surya-ocr-2) | 2025–26 | Datalab license (free < revenue cap) | 90+ langs incl. FA | yes, 1.4 GB | PHTD real hold-out CER 0.540 (`REPORT_dataset_and_bakeoff.md`); not yet scored on gold | **Shortlist**: same family as our layout model |
| 4 | Surya-OCR-2 Persian handwriting 369k fine-tune | [HF](https://huggingface.co/Reza2kn/surya-ocr-2-persian-handwriting-369k) | 2026 | open | FA handwriting (page) | yes | PHTD hold-out CER 0.745 vs 0.540 base | Rejected: worse than its own base |
| 5 | "Bina" (Surya + Persian LoRA, step 8000) | [Triple Threat leaderboard](https://reza2kn-persianocr-triplethreat.hf.space/) | 2026 | unclear | FA printed + handwritten pages | probably | Leaderboard top entry, WER 9.84% on a blended score (weak: page could not be opened to see the handwriting split) | Watch: verify the handwriting split before testing |
| 6 | **Kraken PP-OCRv6 medium** | [HF](https://huggingface.co/small-models-for-glam/kraken-ppocrv6-medium) | 2026-09 | check card | multilingual incl. FA; trained partly on synthetic FA lines | yes, CPU, CTC (fast) | Model card: FA CER 4.66%, WER 20.46% on 990 lines (test set likely printed/synthetic: unverified) | **Shortlist**: line recognizer, cheap; candidate per-word alternatives from CTC |
| 7 | **PaddleOCR-VL** (0.9B) | [blog](https://www.spheron.network/blog/best-open-source-ocr-vlm-self-host-gpu-cloud-2026/) | 2025–26 | Apache-2.0 | multilingual incl. AR | yes | Weak: blog rates handwriting "Good", no FA numbers | **Shortlist** (3rd): different decoder; ensemble partner |
| 8 | Qari-OCR VL-2B | [HF](https://huggingface.co/NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct), [paper](https://arxiv.org/pdf/2506.02295) | 2025 | Apache-2.0 | **AR** printed | yes | **AR**: strongest hallucination pattern (62.4% of errors) in a 2026 [cross-dataset study](https://arxiv.org/html/2608.22366v1) | Skip: hallucination is the failure we can least afford |
| 9 | CRNN on cleaned PHTD (CER-HV) | [arXiv 2601.16713](https://arxiv.org/html/2601.16713v3) | 2026 | paper | FA handwriting, line level | yes | PHTD CER 11.3% (CRNN); releases cleaned PHTD line splits | Use PHTD splits as a 2nd eval set + training data |
| 10 | Handwriting data: MR3z4 (216k real rows), fa-en-ar synthetic, Khayyam (44k words, 400 writers) | [MR3z4](https://huggingface.co/datasets/MR3z4/persian-handwriting-ocr), [synthetic](https://huggingface.co/datasets/saeid1999/fa-en-ar-handwritten-ocr-v1), [Khayyam](https://arxiv.org/pdf/2406.01025) | 2024–26 | NC-SA / open / research | FA handwriting | — | datasets | LoRA fine-tune of #1 (ladder step 5); MR3z4 is non-commercial |
| 11 | Local Persian corrector: ParsBERT MLM | [hooshvare/parsbert](https://github.com/hooshvare/parsbert) | 2020 (base) | Apache-2.0 | FA text | yes, CPU | No published OCR-correction numbers | Local alternative to LLM correction (step 4) |

## Measured on the 20-page subset (48 lines), OCR text only

| Variant | CER | WER | Word acc |
|---|---|---|---|
| Baseline (#1) | 0.3493 | 0.7671 | 23.29% |
| + repetition filter (drop lines with a char ×8+) | 0.3493 | 0.7671 | 23.29% (no effect: metric already caps per-line error) |
| + LLM text correction (Claude, text only, conservative) | **0.3112** | **0.6484** | **35.16%** |

LLM edits: 33 total, 24 helped, 9 neutral, 0 hurt (`htr_eval/experiments/llm_correction_subset20.json`).
Caveat: one hand-run pass by one model on 48 lines; it must be repeated as an automated step before it counts.
