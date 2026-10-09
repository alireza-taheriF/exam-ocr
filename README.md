# Exam OCR / تبدیل برگه آزمون

**English:** A Python pipeline and FastAPI service for converting Persian exam papers (scanned pages or mobile photos) into structured Word documents with OCR text and embedded crops for tables, figures, and messy handwriting.

**فارسی:** سامانه‌ای برای پردازش برگه‌های آزمون فارسی (اسکن یا عکس موبایل) و تولید فایل Word شامل متن OCR و تصاویر کراپ‌شده برای جدول، شکل و دست‌نویس‌های نامرتب.

---

## Project structure / ساختار پروژه

```
exam-ocr/
├── main.py              # FastAPI application
├── static/
│   └── index.html       # Persian upload UI
├── pipeline/
│   ├── preprocess.py    # Image preprocessing
│   ├── layout.py        # Line-level layout (+ OpenCV fallback)
│   ├── ocr.py           # Persian OCR via VLM API
│   ├── crop.py          # Smart cropping for non-OCR blocks
│   ├── normalizer.py    # Shared Persian normalization
│   └── export.py        # DOCX export (RTL)
├── eval/                # Evaluation harness (split / metrics / baseline)
│   ├── split.py
│   ├── normalize.py
│   ├── metrics.py
│   ├── run_baseline.py
│   └── splits/          # train / val / gold_test manifests
├── dataset/
│   ├── gold.jsonl       # Hand-verified labeled line crops
│   └── lines_v3/        # Line crop images
├── uploads/             # Uploaded input files
├── outputs/             # Generated DOCX + eval artifacts
└── logs/
    └── errors.log       # Error log (created at runtime)
```

---

## Installation / نصب

**Python 3.10, 3.11, or 3.12 is required.** Python 3.13+ is not supported by `surya-ocr` 0.6.x with current dependencies.

```bash
cd exam-ocr
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Requirements:** PyTorch and Surya OCR models (downloaded automatically on first OCR run into `.cache/huggingface/`).

---

## Run the server / اجرای سرور

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open in your browser:

- Upload UI: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Hand-label sealed new_gold / برچسب‌زنی مجموعهٔ مهرشده

A **local-only** correction UI for the sealed 300-line set under `eval/new_gold/`
(excluded from every training mix by path + SHA-256). Binds to `127.0.0.1` only.

```bash
# Cache OCR preds for both local backends (Qwen3 via transformers; Qwen2.5 via LM Studio)
python -m labeling.precompute_preds --models qwen3
python -m labeling.precompute_preds --models qwen25   # needs LM Studio on :1234

# Label (≈50–75 min at 10–15 s/line; ~315 presentations with 5% repeats)
python -m labeling.ui --queue new_gold
# → http://127.0.0.1:8765/

# Score models on your corrections (body vs form_field, blind vs prefilled, …)
python -m eval.new_gold_report
```

Details: [`labeling/README.md`](labeling/README.md). Do **not** train on these crops.

## Evaluation harness / ارزیابی

Clean hand-labeled data lives in **`dataset/gold.jsonl`** (240 line crops; 168 with
non-empty text), with images under `dataset/lines_v3/` and page links via
`source_image` / `source_document`. This set is separate from the old broken
page→line alignment outputs (`dataset/lines.jsonl`, `dataset/dataset.jsonl`).

**Primary metric:** mean **sentence-level similarity** with partial credit
(after `eval/normalize.py`):

```
sentence_similarity = clip(1 - levenshtein(pred, ref) / max(len(ref), 1), 0, 1)
```

Also reported: % units with similarity ≥ 0.80, exact-match rate, CER, WER, word accuracy.
Because gold labels are line crops with sparse sentence punctuation, the harness
**falls back to line-level units** (newline-separated) and states this in the report.
`gold_test` is hold-out only — never use it for training, tuning, or thresholding.

```bash
source .venv/bin/activate

# 1) Document-level split (default) → eval/splits_by_document/
python -m eval.split --seed 42
# Optional page-level split (leaks documents across splits):
# python -m eval.split --split-by page --out-dir eval/splits

# 2) Unit tests for metrics / normalize / split
pytest eval/tests -q

# 3) Recognition baseline (default: line_crops + qwen_local on document split)
python -m eval.run_baseline --splits val gold_test

# Fine-tuned LoRA adapter (after Phase C training):
# QWEN_LORA_PATH=models/qwen3-vl-fa-exam-lora-v1 python -m eval.run_baseline --splits val

# 4) Layout debugging only (not the primary metric):
python -m eval.run_baseline --mode page --ocr-backend surya --compare-layout --splits val
python -m eval.run_baseline --mode page --debug-overlays 10 --overlays-only
```

Primary workflow is **line_crops recognition** with `qwen_local` on
`eval/splits_by_document`. Page-mode scoring matches gold lines to detected
blocks for layout diagnosis only. Outputs land in `outputs/eval/`.

**Note:** keep `transformers` in the range pinned by `requirements.txt`
(`>=4.41,<4.46`). Newer major versions break Surya recognition/ordering imports.

---

## Testing / تست

### 1) Quick health check

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### 2) Upload via browser

1. Open `http://127.0.0.1:8000/`
2. Choose a real exam scan/photo (`jpg`, `png`, `tiff`, or `pdf`)
3. Click **شروع پردازش**
4. Download the Word file from the result link

> **Note:** A synthetic test image may return HTTP 500 (`OCR failed completely`) because it contains no real Persian text.

### 3) Upload via curl

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@/path/to/exam.jpg;type=image/jpeg;filename=exam.jpg"
```

Example success response:

```json
{
  "job_id": "371c809b-9cc1-4e83-8291-70b1bb078a50",
  "status": "done",
  "output_file": "outputs/result_371c809b-9cc1-4e83-8291-70b1bb078a50.docx",
  "block_count": 12,
  "text_blocks": 9,
  "cropped_blocks": 3
}
```

### 4) Check status and download

```bash
JOB_ID="<paste job_id here>"

curl "http://127.0.0.1:8000/status/${JOB_ID}"

curl -OJ "http://127.0.0.1:8000/download/${JOB_ID}"
```

### 5) Verify imports locally

```bash
python -c "from main import app; from pipeline.ocr import run_ocr; print('OK')"
```

### 6) Error log

If something fails, inspect:

```bash
cat logs/errors.log
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload file and run full pipeline |
| `GET` | `/download/{job_id}` | Download generated DOCX |
| `GET` | `/status/{job_id}` | Check job status |
| `GET` | `/health` | Health check |

**Example upload (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@uploads/exam.jpg"
```

**Example response:**

```json
{
  "job_id": "uuid",
  "status": "done",
  "output_file": "outputs/result_uuid.docx",
  "block_count": 12,
  "text_blocks": 9,
  "cropped_blocks": 3
}
```

---

## Pipeline diagram / نمودار pipeline

```
┌─────────────┐
│ Upload file │  jpg / png / tiff / pdf
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  preprocess.py  │  grayscale → deskew → denoise → threshold → CLAHE
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   layout.py     │  detect blocks: text / table / figure / handwriting
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│    ocr.py       │  Surya OCR (fa) on text blocks
│    crop.py      │  crop tables, figures, low-confidence handwriting
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│   export.py     │  RTL Word document (.docx)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Download DOCX   │  GET /download/{job_id}
└─────────────────┘
```

**Block routing:**

- **Printed text** → OCR → text paragraph in Word
- **Table / figure** → crop → embedded image in Word
- **Messy handwriting** (confidence < 0.6) → crop → embedded image in Word

Blocks are ordered **top-to-bottom, right-to-left** (Persian reading order).

---

## Programmatic usage / استفاده برنامه‌نویسی

```python
import cv2
from pipeline.preprocess import preprocess_image
from pipeline.layout import detect_blocks
from pipeline.ocr import run_ocr
from pipeline.export import export_to_word

original = cv2.imread("uploads/exam.jpg")
processed = preprocess_image("uploads/exam.jpg")
blocks = detect_blocks(processed)
blocks = run_ocr(original, blocks)
export_to_word(blocks, "outputs/exam.docx")
```

---

## Error handling / مدیریت خطا

- Invalid or unreadable uploads → HTTP `400`
- Complete OCR failure (no text extracted) → HTTP `500`
- All errors are logged to `logs/errors.log`

---

## Notes / نکات

- First run downloads Surya OCR models (may take a few minutes).
- PDF files are rendered page-by-page before processing.
- Recommended input: clear scans or well-lit mobile photos of exam pages.
