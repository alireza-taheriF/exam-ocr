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
│   ├── layout.py        # Block detection (text, table, figure, handwriting)
│   ├── ocr.py           # Persian OCR via Surya
│   ├── crop.py          # Smart cropping for non-OCR blocks
│   └── export.py        # DOCX export (RTL)
├── uploads/             # Uploaded input files
├── outputs/             # Generated DOCX files
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

### Local OCR engine / موتور OCR محلی

Text is read **on this machine only**: Surya finds the layout and the lines, and every line is read by
`mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0` (best of the engines measured in `htr_eval/REPORT.md`).
It needs `transformers>=5`, which conflicts with the app's `transformers<4.46`, so it runs in a second venv
that the app starts as a worker process:

```bash
python3.12 -m venv .venv-ocr
.venv-ocr/bin/pip install torch "transformers>=5" accelerate pillow
.venv-ocr/bin/python -c "from huggingface_hub import snapshot_download as d; d('mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0')"
export OCR_PYTHON=$PWD/.venv-ocr/bin/python   # default: ./.venv/bin/python
```

Remote vision APIs are **off**: set `OCR_ALLOW_REMOTE=1` to opt in (block crops are then uploaded to `VLM_API_BASE`).
LLM post-correction (Ollama) did not improve WER in our tests, so it is not part of the pipeline.

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
