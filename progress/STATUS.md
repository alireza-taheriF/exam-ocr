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
