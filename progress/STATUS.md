# STATUS

**Phase:** 1 done (pending commit/push) → next: Phase 2 (quick wins, no training)
**Done in Phase 1**
- scripts/build_dev_set.py → dataset/dev_todo.jsonl (400 crops from val_v3; val has only 5 docs with ≥20 lines, so dev covers 5 docs; web doc has 44 lines).
- Labeled by reading contact sheets (10 crops/sheet, scripts/make_label_sheets.py + add_labels.py) → dataset/dev_claude.jsonl.
  - 400 rows: 132 scorable, 268 illegible/fragments (pen strokes, specks, ink blobs, multi-line fragments).
- QC (dataset/dev_qc_claude.jsonl, 40 random rows, re-read): CER between passes on the 12 lines labeled both times = 0.000, BUT this is not an independent measure (same session, labels still in context). More informative: 4 of 16 rows I first marked illegible were readable on re-read; those 4 labels were adopted (labeler=claude-code-qc). Expect ~10-15% of "illegible" rows may be readable.
- scripts/run_ocr_preds.py (inference, resumable, dynamic tokens/rep-penalty/adapter flags) and scripts/eval_ocr.py (CER/WER/acc overall, per doc, per width; --min-width).
- Baseline on dev: CER 0.4928, acc 50.72%, WER 0.758, exact 16.7% (132 lines). Degenerate repeats ("۱۰۰۰۰…", "1111…") visible → phase 2 targets.
**Run model:** `HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-qwen/bin/python scripts/run_ocr_preds.py --data dataset/dev_claude.jsonl --pred dataset/preds/<name>.jsonl [flags]`; score: `.venv/bin/python scripts/eval_ocr.py --gold dataset/dev_claude.jsonl --pred ... --report dataset/reports/<name>.json`
**Caveats**
- dev is small (132 scorable lines, 5 docs) → differences < ~1.5 CER points are noise.
- Earlier-session finding: LoRA on 855 Claude labels scored 60.7% on gold vs base 63.1% (rejected); opencv segmentation fragments are a bottleneck. Phase 5 must beat base on dev before being kept.
**Next:** commit+push Phase 1; Phase 2 items 1-5 each measured on dev; one gold run at end of phase.

**Known failing tests (not our change):** 4 tests in test_pipeline.py (PipelineIntegrationTest: full_pipeline_with_mock_vlm, invalid_api_key_falls_back_to_crop, layout_detects_blocks, layout_fallback_on_blank_page) fail with `ImportError: find_pruneable_heads_and_indices` — surya 0.6.13 vs the newer transformers in .venv. They fail on HEAD before any Phase-1 change (this commit only adds scripts/data). Not being fixed now.

## Update — user corrections (after Phase 1 report)
- Item 5 done (failing tests noted above).
- Item 3 DONE: blind full-res (4 crops/sheet) transcription of all 240 gold crops (paths only; gold text not opened) → dataset/gold_selfcheck_claude.jsonl. Never used for training/tuning.
  - Result vs human gold: **CER 0.2626** (acc 73.7%) on 168 scorable rows → ABOVE the 0.10 threshold; STOP before Phase 4 and tell user.
  - Per doc CER: مقدمات 0.158, CamScanner 0.310, ریاضی 0.396, اصول حسابداری 0.461. Report: dataset/reports/claude_labeler_vs_gold.json
  - Error sources: (a) 35 of 168 gold rows I left empty (662/4467 gold chars): 16 are ≤4 chars (ص ۵, فصل ۴, ۰, شرح) — tiny/blurry crops gold annotator labeled; incl. ≥1 clear mistake (one readable line skipped). (b) on 133 rows both labeled: CER 0.134 — conventions (I add space before ':' and '=', arrows '←', Persian digits/commas), 2 gold rows keyboard-garbled (unfixable), digits ۲۹ vs ۲۶ misreads, partial-line crops.
- Items 1, 2, 4: NOT yet done (dev relabel of 268 illegible at native res; dev ≥300 scorable from width>150 + bootstrap CI; LoRA failure diagnosis) — pending user decision.
- Phase 2 dev sweep (132 lines, CER): base 0.4928 | dyn_tokens15 0.5007 | rep1.05 0.4673 | rep1.10 0.4789 | rep1.15 0.4896 | dyn15+rep1.05 0.4752 | dyn15+rep1.10 0.4868. Differences < noise; none proven.
