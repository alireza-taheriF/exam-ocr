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

## Update 3 — labeler re-measure (user items 1-5 of the 2nd message)
- Metric: training/eval/metrics.py::lenient_normalize (symmetric: arrows, comma variants, spaces around punctuation); eval_ocr.py reports strict+lenient+bootstrap CI95. All earlier numbers recomputed in dataset/reports/v2/.
  - base@gold strict .3770 / lenient .3780; base@gold_v1.1 .3714 / .3724
  - dev v1 base .4928/.4788 ; rp1.05 .4673/.4524 (CIs overlap → no proven gain)
- gold_v1.1.jsonl: only row 201 (QWERTY→Persian) fixed; row 200 ("Scanned by CamScanner" typed on Persian layout) NOT fixed because the inverse mapping cannot recover the leading shift-C deterministically → left as is. Log: dataset/gold_v1.1_changes.json.
- docs/transcription_rules.md written (general rules only).
- Labeler round 2 (context sheets: 3 crops/image, enlarged crop + page context, scripts/make_context_sheets.py), single re-measure as instructed:
  - gold_v1.1: strict **0.1081** [0.091,0.127], lenient **0.0889** [0.072,0.108]; gold.jsonl: 0.1209 / 0.1023. ≤0.15 → passes the user's gate.
  - CAVEAT (must be read with the number): before round 2 I had printed ~40 gold strings (worst-error list after round 1: e.g. «ص ۵», «فصل ۴», «شرح», «x : 0», ...). The decision to transcribe tiny crops was user-directed, but my round-2 readings of those crops were not blind to those strings. True blind labeler CER is likely higher than 0.108; I estimate it unknowable from this run. A clean check would need a fresh blind sample (e.g. dev crops double-labeled), not another pass on gold.
- Next (per user): relabel dev with the new method (400 existing + 268 illegible; extend to ≥300 scorable from val_v3 width>150), bootstrap CIs, re-confirm rep_penalty=1.05, LoRA-failure diagnosis (items 4 a–e) before Phase 5.
