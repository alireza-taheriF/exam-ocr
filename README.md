# Exam OCR / تبدیل برگه آزمون

**English:** A Python pipeline and FastAPI service for converting Persian exam papers (scanned pages or mobile photos) into structured Word documents with OCR text and embedded crops for tables, figures, and messy handwriting.

**فارسی:** سامانه‌ای برای پردازش برگه‌های آزمون فارسی (اسکن یا عکس موبایل) و تولید فایل Word شامل متن OCR و تصاویر کراپ‌شده برای جدول، شکل و دست‌نویس‌های نامرتب.

---

## Student editing app / سامانهٔ ویرایش دانشجو

دانشجو برگهٔ اسکن‌شده را بارگذاری می‌کند؛ هر خط تشخیص‌داده‌شده به‌صورت **تصویر خط + باکس متن قابل‌ویرایش**
نمایش داده می‌شود. ویرایش‌ها خودکار و **append-only** ذخیره می‌شوند (متن اولیهٔ OCR و همهٔ نسخه‌ها برای مدرس
قابل‌مشاهده است). بعد از «تأیید نهایی» سند قفل می‌شود. اصلاحات تأییدشده:

1. **فوراً** وارد حافظهٔ اصلاحات می‌شوند: اصلاحی که چند دانشجوی مستقل تکرار کرده باشند روی خروجی‌های بعدی خودکار اعمال می‌شود.
2. **دوره‌ای** با `scripts/export_corrections.py` برای fine-tune روی Colab خروجی گرفته می‌شوند.

پلن کامل: [`PLAN_90_percent_feedback_loop.md`](PLAN_90_percent_feedback_loop.md)

### اجرای دمو (بدون مدل، چند ثانیه)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-app.txt
OCR_ENGINE=fake SEGMENTER=opencv HOLDOUT_FRACTION=0 MEMORY_MIN_AUTHORS=2 uvicorn main:app --port 8000
```

- دانشجو: <http://127.0.0.1:8000/>
- پنل یادگیری/مدرس: <http://127.0.0.1:8000/admin>

موتور `fake` برای همهٔ خطوط یک جملهٔ ثابت با یک غلط عمدی («هسادجاری») برمی‌گرداند. برای نشان دادن یادگیری:
با دو نام مختلف برگه را بارگذاری کنید، تیک رضایت را بزنید، «هسادجاری» را به «حسابداری» اصلاح و تأیید کنید؛
بارگذاری سوم آن را خودکار اصلاح‌شده (با برچسب سبز «اصلاح خودکار») نشان می‌دهد.

### اجرا با مدل واقعی (مک، CPU/MPS)

```bash
python3.12 -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
OCR_ENGINE=qwen_local SEGMENTER=opencv uvicorn main:app --port 8000
```

مدل `mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0` (۴٫۳ گیگ) بار اول دانلود می‌شود؛ روی M2 حدود
۱٫۴ ثانیه برای هر خط (اندازه‌گیری‌شده در گزارش bake-off). Surya به‌دلیل ناسازگاری `transformers` در این محیط
نصب نمی‌شود؛ تقسیم خطوط با OpenCV انجام می‌شود.

### API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/documents` | بارگذاری (`file`, `author`, `course`, `consent`) — پردازش در پس‌زمینه |
| `GET` | `/api/documents/{id}` | وضعیت + خطوط (متن فعلی، اطمینان، اصلاحات خودکار) |
| `PUT` | `/api/lines/{id}` | ذخیرهٔ ویرایش (`text`, `author`, `edit_ms`) — نسخهٔ جدید |
| `GET` | `/api/lines/{id}/history` | خروجی خام مدل + همهٔ نسخه‌های ویرایش |
| `GET` | `/api/lines/{id}/crop` | تصویر خط |
| `POST` | `/api/documents/{id}/confirm` | تأیید نهایی و قفل |
| `GET` | `/api/documents/{id}/export?format=docx\|txt` | خروجی از متن ویرایش‌شده |
| `GET` | `/api/stats` | دقت پس‌از‌ویرایش به تفکیک نسخهٔ مدل، قواعد یادگرفته، ویرایش‌های قرنطینه |
| `GET` | `/health` | وضعیت |

### خروجی برای آموزش

```bash
python scripts/export_corrections.py --out exports/corrections_v1
```

فقط اسناد تأییدشده با رضایت، و فقط اصلاحات «قابل‌اعتماد» (`learning/label_trust.py`) صادر می‌شوند؛
اسناد hold-out جداگانه در `holdout.jsonl` برای ارزیابی می‌آیند.

### تست

```bash
pytest tests learning pipeline/tests training/eval/tests
```

---

## Project structure / ساختار پروژه

```
exam-ocr/
├── main.py                 # FastAPI app (student editor + admin API)
├── app/store.py            # SQLite: documents, lines, predictions, append-only edits
├── app/service.py          # render → segment → OCR → correction memory → DB
├── pipeline/segment.py     # line segmentation (Surya or OpenCV)
├── pipeline/engines.py     # OCR engines: qwen_local | vlm | fake
├── learning/               # label trust + correction memory (fast learning loop)
├── static/index.html       # student editor (RTL)
├── static/admin.html       # learning / instructor panel
├── scripts/export_corrections.py
├── pipeline/ training/ scripts/ dataset/   # OCR pipeline, eval, dataset tooling
└── data/                   # runtime DB, uploads, crops (gitignored)
```

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
