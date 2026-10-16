# exam-ocr — notes for Claude

Persian exam-sheet OCR: FastAPI (`main.py`) + pipeline (`pipeline/`) + training/eval (`training/`, `scripts/`).

- Active plan: `PLAN_90_percent_feedback_loop.md`. Tick items and record measured numbers there as work lands.
- Before any change that can affect recognized text, follow `.claude/skills/ocr-eval-gate`.
- Corrections / learning loop work follows `.claude/skills/feedback-learning`.
- Persian text and RTL rules: `.claude/skills/persian-text`.
- Dataset images and models are gitignored (`dataset/lines_v3/`, `models/`); in cloud sessions, use the `fake` OCR engine for tests and do not claim accuracy numbers.
- Tests: `pytest` from repo root.
