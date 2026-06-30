# Exam OCR — نقشه راه تا دقت ۹۵٪+

> **این فایل منبع حقیقت (Source of Truth) پروژه است.**
> هر بار قبل از شروع کار، این فایل را بخوان. هر کار انجام‌شده را تیک بزن، تاریخ و توضیح کوتاه بنویس.
> هدف نهایی: **CER ≤ ۵٪** (معادل ~۹۵٪ دقت کاراکتری) روی مجموعه تست hold-out برگه‌های امتحانی فارسی.

---

## وضعیت کلی

| شاخص | مقدار فعلی | هدف |
|------|-----------|-----|
| تصاویر خام | ۱۲۶ | ۲۰۰۰+ |
| برچسب‌های معتبر | ۳ | ۱۵۰۰+ |
| مدل OCR تولید | VLM API (بدون vision درست) | TrOCR fine-tuned + VLM fallback |
| ارزیابی خودکار | ✅ `training/eval/` | ✅ CER/WER روی test set |
| پیش‌پردازش در API | ❌ | ✅ |
| دقت تخمینی فعلی | **۰٪** (CER=1.0 — VLM بدون vision) | **≥ ۹۵٪** |

**آخرین به‌روزرسانی:** ۲۰۲۶-۰۶-۲۳ — فاز ۰ کامل شد.

| فاز فعلی | وضعیت |
|---------|--------|
| فاز ۰ | ✅ کامل |
| فاز ۱ | 🔄 در حال اجرا (crawler + auto_labeler) |

---

## تعریف «دقت ۹۵٪»

برای این پروژه معیار اصلی:

```
CER (Character Error Rate) = (جایگزینی + حذف + درج) / تعداد کل کاراکترهای ground truth
هدف: CER ≤ 0.05  →  دقت کاراکتری ≥ 95%
```

معیارهای ثانویه:
- **WER ≤ ۱۰٪** (دقت واژه‌ای ~۹۰٪ — برای دست‌نویس امتحان قابل قبول)
- **Block accuracy ≥ ۹۰٪** (بلوک‌هایی که متن قابل قبول دارند)
- **Illegible rate ≤ ۵٪** (بلوک‌هایی که فقط crop می‌شوند)

ارزیابی روی **حداقل ۱۰۰ تصویر hold-out** که در آموزش استفاده نشده‌اند.

---

## معماری هدف

```
آپلود برگه
    ↓
پیش‌پردازش (deskew, denoise, CLAHE)
    ↓
Surya Layout + Reading Order
    ↓
برای هر بلوک متن/لیست:
    ├─ TrOCR fine-tuned (لوکال) ── confidence ≥ 0.7 → متن نهایی
    ├─ confidence < 0.7 → VLM fallback (رایگان)
    └─ هر دو ناموفق → crop تصویر در Word
    ↓
export RTL DOCX
```

---

## ساختار پوشه‌های مدل (مهم — بعد از Colab اینجا بذار)

```
exam-ocr/
├── models/                          # ← مدل‌های آموزش‌داده‌شده (از Colab بیار اینجا)
│   ├── trocr-fa-exam-v1/            # مدل اصلی OCR خط فارسی
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin        # یا model.safetensors
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   ├── merges.txt               # اگر BPE
│   │   └── training_meta.json       # epoch, CER, تاریخ آموزش
│   └── trocr-fa-exam-v2/            # نسخه‌های بعدی
├── .cache/huggingface/hub/          # مدل‌های Surya (خودکار)
├── dataset/
│   ├── raw_images/                  # تصاویر کامل صفحه
│   ├── lines/                       # کراپ خطوط (برای آموزش TrOCR)
│   ├── manifest.jsonl
│   ├── dataset.jsonl                # برچسب سطح تصویر
│   ├── lines.jsonl                  # برچسب سطح خط: {image, text, source_image}
│   ├── train.jsonl                  # split آموزش
│   ├── val.jsonl                    # split اعتبارسنجی
│   ├── test.jsonl                   # split تست (hold-out — لمس نکن!)
│   └── label_errors.jsonl
├── training/
│   ├── colab/
│   │   ├── train_trocr_fa.ipynb     # نوت‌بوک Colab (ساخته می‌شود)
│   │   └── export_for_inference.py
│   └── eval/
│       ├── metrics.py               # CER/WER
│       └── evaluate.py              # ارزیابی end-to-end
└── ROADMAP.md                       # همین فایل
```

### وقتی مدل را از Colab آوردی

1. فایل‌های export‌شده را در `models/trocr-fa-exam-v1/` کپی کن.
2. در `.env` اضافه کن:
   ```
   TROCR_MODEL_PATH=models/trocr-fa-exam-v1
   OCR_ENGINE=hybrid          # hybrid | trocr | vlm
   TROCR_CONFIDENCE_THRESHOLD=0.7
   ```
3. سرور را ری‌استارت کن و `python training/eval/evaluate.py` را اجرا کن.

---

# فاز ۰ — زیرساخت و خط مبنا (Baseline)
> **GPU لازم نیست** | تخمین: ۲–۳ روز

- [x] **۰.۱** — اصلاح `.env.example` و مستندسازی تنظیمات vision
  - توضیح: `VLM_API_*` باید به مدل vision واقعی اشاره کند، نه `deepseek-chat`
  - انجام‌شده: `.env.example` بازنویسی شد؛ پیش‌فرض به FreeLLMAPI/Nemotron VL تغییر کرد
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۰.۲** — تست health-check کامل pipeline روی ۵ تصویر نمونه
  - فایل‌ها: `scripts/baseline_health_check.py`, `logs/baseline_report.txt`
  - نتیجه: ۵/۵ تصویر layout OK؛ با OCR همه بلوک‌ها crop شدند (deepseek-chat vision ندارد)
  - انجام‌شده: اسکریپت ساخته و اجرا شد
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۰.۳** — ساخت `training/eval/metrics.py` (توابع CER و WER)
  - انجام‌شده: `cer`, `wer`, `normalize_text`, `accuracy_from_cer`
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۰.۴** — ساخت `training/eval/evaluate.py` (ارزیابی روی `dataset/dataset.jsonl`)
  - انجام‌شده: ارزیابی با فیلتر هالوسینیشن + خروجی JSON
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۰.۵** — اندازه‌گیری baseline فعلی و ثبت CER در این فایل
  - Baseline CER: **1.0** (دقت ۰٪) — ۳ نمونه معتبر، ۰ بلوک متنی
  - علت: `.env` فعلی `deepseek-chat` است (text-only) → همه OCRها fail → crop
  - گزارش: `logs/eval_report.json`
  - اقدام بعدی (فاز ۱): تنظیم vision API + برچسب‌گذاری مجدد
  - انجام‌شده: ارزیابی اجرا شد
  - تاریخ: ۲۰۲۶-۰۶-۲۳

---

# فاز ۱ — جمع‌آوری و برچسب‌گذاری داده
> **GPU لازم نیست** (فقط API رایگان vision) | تخمین: ۱–۲ هفته

## ۱.۱ تصاویر بیشتر

- [x] **۱.۱.۱** — اجرای crawler برای حداقل ۵۰۰ تصویر جدید
  ```bash
  python crawler.py --target 500
  ```
  - انجام‌شده: `--target` به crawler اضافه شد؛ crawler در پس‌زمینه اجرا شد (`logs/crawler.log`)
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۱.۱.۲** — فیلتر کیفیت: حذف تصاویر کوچک‌تر از ۴۰۰px، تار، یا بدون متن فارسی
  - اسکریپت: `scripts/filter_dataset.py`
  - نتیجه: ۷ تصویر رد شد، ۱۲۶ باقی ماند
  - انجام‌شده: اجرا شد
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [ ] **۱.۱.۳** — هدف نهایی تصاویر خام: **≥ ۲۰۰۰** تصویر
  - تعداد فعلی: ۱۲۶ (crawler در حال اجرا → هدف ۵۰۰)
  - انجام‌شده: در حال اجرا
  - تاریخ: ۲۰۲۶-۰۶-۲۳

## ۱.۲ برچسب‌گذاری خودکار (سطح تصویر)

- [x] **۱.۲.۱** — راه‌اندازی FreeLLMAPI یا OpenRouter vision
  - انجام‌شده: OpenRouter مستقیم (`OPENROUTER_API_KEY`) + `scripts/sync_vlm_env.py`
  - تست: ۳/۵ تصویر در limit=5 با موفقیت لیبل شد
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [ ] **۱.۲.۲** — اجرای auto_labeler روی همه تصاویر
  ```bash
  python auto_labeler.py --provider openai --resume
  ```
  - انجام‌شده: در پس‌زمینه اجرا شد (`logs/auto_labeler.log`)
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۱.۲.۳** — فیلتر هالوسینیشن: حذف نمونه‌هایی با تکرار بیش از حد یا طول غیرعادی
  - انجام‌شده: `training/eval/label_quality.py` + `scripts/clean_labels.py` (۵ برچسب بد حذف شد)
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [ ] **۱.۲.۴** — هدف: **≥ ۱۵۰۰** برچسب معتبر در `dataset/dataset.jsonl`
  - تعداد فعلی: ۳ معتبر / ۶ کل (auto_labeler در حال اجرا)
  - انجام‌شده: در حال اجرا
  - تاریخ: ۲۰۲۶-۰۶-۲۳

## ۱.۳ استخراج خطوط (برای آموزش TrOCR)

- [x] **۱.۳.۱** — ساخت `scripts/extract_lines.py`
  - انجام‌شده: Surya line detection + crop
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۱.۳.۲** — تراز خطوط با متن برچسب (alignment)
  - انجام‌شده: min-count alignment + `dataset/lines_mismatch.jsonl`
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [ ] **۱.۳.۳** — هدف: **≥ ۸۰۰۰** نمونه خط در `lines.jsonl`
  - تعداد فعلی: ۵
  - انجام‌شده: بعد از اتمام auto_labeler دوباره اجرا شود
  - تاریخ: ۲۰۲۶-۰۶-۲۳

## ۱.۴ دیتاست‌های عمومی فارسی (تقویت آموزش)

- [x] **۱.۴.۱** — اسکریپت دانلود **Persian-OCR-230k**
  - مسیر: `scripts/download_external_datasets.py` → `dataset/external/persian-ocr-230k/`
  - اجرا: `pip install datasets && python scripts/download_external_datasets.py --dataset persian-ocr-230k --limit 1000 --merge`
  - انجام‌شده: اسکریپت ساخته شد (دانلود هنوز اجرا نشده)
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [ ] **۱.۴.۲** — دانلود **Iranshahr** و **Sadri** (دست‌نویس کلمه‌ای)
  - مسیر: `dataset/external/iranshahr/`, `dataset/external/sadri/`
  - یادداشت: نیاز به دانلود دستی از مخازن دانشگاهی — در فاز ۳ Colab اضافه می‌شود
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۱.۴.۳** — ادغام دیتاست‌ها در `dataset/all_train.jsonl`
  - انجام‌شده: بعد از دانلود خارجی
  - تاریخ: —

## ۱.۵ تقسیم train/val/test

- [x] **۱.۵.۱** — ساخت `scripts/split_dataset.py`
  - انجام‌شده: ۷۰/۱۵/۱۵ split
  - تاریخ: ۲۰۲۶-۰۶-۲۳

- [x] **۱.۵.۲** — ثبت آمار split در این فایل
  - train: ۳ | val: ۰ | test: ۲ (موقت — بعد از extract_lines کامل به‌روز شود)
  - انجام‌شده: split اولیه روی ۵ خط
  - تاریخ: ۲۰۲۶-۰۶-۲۳

---

# فاز ۲ — بهبود Pipeline (بدون آموزش مدل)
> **GPU لازم نیست** | تخمین: ۳–۵ روز

- [ ] **۲.۱** — اتصال `pipeline/preprocess.py` به `main.py`
  - deskew → denoise → CLAHE قبل از layout
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۲.۲** — تست تاثیر preprocess روی ۲۰ تصویر و مقایسه CER
  - CER قبل: ___ | CER بعد: ___
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۲.۳** — اصلاح `pipeline/ocr.py` برای پشتیبانی چند موتور
  - `OCR_ENGINE=vlm|trocr|hybrid`
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۲.۴** — بهبود prompt و retry logic در VLM
  - temperature=0, max_retries=2, validation خروجی فارسی
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۲.۵** — اضافه کردن confidence score به خروجی OCR
  - برای TrOCR: logits/entropy؛ برای VLM: heuristics (طول، نسبت حروف فارسی)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۲.۶** — به‌روزرسانی README با معماری واقعی
  - انجام‌شده: —
  - تاریخ: —

---

# فاز ۳ — آموزش TrOCR روی Colab
> **🖥️ GPU لازم است — Google Colab Pro/T4 یا A100** | تخمین: ۱ هفته

## پیش‌نیازها (قبل از Colab)

- [ ] **۳.۰.۱** — فاز ۱ کامل شده باشد (حداقل ۵۰۰۰ خط در `lines.jsonl`)
- [ ] **۳.۰.۲** — `train.jsonl` و `val.jsonl` آماده باشند
- [ ] **۳.۰.۳** — نوت‌بوک `training/colab/train_trocr_fa.ipynb` ساخته شده باشد

## مراحل Colab (کاربر انجام می‌دهد)

- [ ] **۳.۱** — آپلود `train.jsonl`, `val.jsonl` و پوشه `dataset/lines/` به Colab
  - روش پیشنهادی: zip کردن و آپلود به Google Drive
  - مسیر در Drive: `MyDrive/exam-ocr/training/`
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۳.۲** — اجرای نوت‌بوک آموزش
  - Base model: `microsoft/trocr-base-handwritten`
  - Tokenizer: `xlm-roberta-base` (پشتیبانی فارسی)
  - Epochs: ۱۵–۳۰ (با early stopping روی val CER)
  - Batch size: ۸ (T4) یا ۱۶ (A100)
  - Learning rate: 5e-5 → 1e-5 (cosine decay)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۳.۳** — مانیتور val CER در حین آموزش
  - هدف میانی: val CER < 0.15 (دقت ~۸۵٪)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۳.۴** — Export مدل نهایی
  ```python
  model.save_pretrained("trocr-fa-exam-v1")
  processor.save_pretrained("trocr-fa-exam-v1")
  ```
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۳.۵** — دانلود و قرار دادن در پروژه
  ```
  exam-ocr/models/trocr-fa-exam-v1/
  ```
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۳.۶** — ثبت نتایج آموزش در `models/trocr-fa-exam-v1/training_meta.json`
  ```json
  {
    "base_model": "microsoft/trocr-base-handwritten",
    "tokenizer": "xlm-roberta-base",
    "epochs": 20,
    "val_cer": 0.12,
    "train_samples": 7000,
    "date": "2026-XX-XX",
    "colab_gpu": "T4"
  }
  ```
  - انجام‌شده: —
  - تاریخ: —

### نکات Colab

| مورد | مقدار |
|------|-------|
| GPU پیشنهادی | T4 (رایگان) یا A100 (Pro) |
| زمان تقریبی | ۴–۱۲ ساعت بسته به حجم داده |
| RAM | ≥ ۱۶ GB |
| دیسک Drive | ≥ ۵ GB |

---

# فاز ۴ — ادغام مدل و Hybrid OCR
> **GPU اختیاری** (inference روی CPU کند است؛ GPU بهتر) | تخمین: ۳–۵ روز

- [ ] **۴.۱** — ساخت `pipeline/trocr_ocr.py`
  - بارگذاری از `TROCR_MODEL_PATH`
  - predict روی crop بلوک
  - برگرداندن text + confidence
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۴.۲** — پیاده‌سازی hybrid در `pipeline/ocr.py`
  ```
  if confidence >= 0.7: use TrOCR
  else: fallback VLM
  else: crop
  ```
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۴.۳** — تست inference لوکال روی ۵۰ تصویر val
  - val CER: ___
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۴.۴** — ارزیابی end-to-end روی test set
  - test CER: ___
  - انجام‌شده: —
  - تاریخ: —

---

# فاز ۵ — حلقه بهبود تا ۹۵٪
> **ترکیب GPU (Colab) و بدون GPU** | تخمین: ۲–۴ هفته

## اگر CER > 0.10 (دقت < ۹۰٪)

- [ ] **۵.۱** — بازبینی دستی ۱۰۰ نمونه خط با بدترین CER
  - شناسایی الگوهای خطا: اعداد، علائم، دست‌خط نامرتب
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۲** — افزودن نمونه‌های خطا به دیتاست آموزش (hard example mining)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۳** — آموزش مجدد v2 روی Colab با داده بیشتر
  - خروجی: `models/trocr-fa-exam-v2/`
  - انجام‌شده: —
  - تاریخ: —

## اگر CER 0.05–0.10 (دقت ۹۰–۹۵٪)

- [ ] **۵.۴** — تنظیم threshold hybrid (0.5–0.8)
  - بهترین threshold: ___
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۵** — fine-tune روی زیرمجموعه دست‌نویس خالص (بدون چاپی)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۶** — data augmentation: چرخش جزئی، blur، نویز، کنتراست
  - انجام‌شده: —
  - تاریخ: —

## اگر CER ≤ 0.05 (دقت ≥ ۹۵٪) ✅

- [ ] **۵.۷** — تست روی ۲۰ برگه امتحان واقعی (خارج از دیتاست)
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۸** — تست سرعت: زمان پردازش هر صفحه
  - هدف: < ۳۰ ثانیه روی CPU، < ۱۰ ثانیه روی GPU
  - انجام‌شده: —
  - تاریخ: —

- [ ] **۵.۹** — مستندسازی نهایی و freeze کردن مدل production
  - مدل نهایی: `models/trocr-fa-exam-v___/`
  - انجام‌شده: —
  - تاریخ: —

---

# فاز ۶ — بهینه‌سازی Production (اختیاری)
> **GPU برای ONNX export مفید است**

- [ ] **۶.۱** — Export به ONNX برای inference سریع‌تر
- [ ] **۶.۲** — Batch inference برای چند بلوک همزمان
- [ ] **۶.۳** — Caching مدل در حافظه (warm start)
- [ ] **۶.۴** — Docker image برای deploy

---

## لاگ پیشرفت

| تاریخ | فاز | کار انجام‌شده | نتیجه |
|-------|-----|--------------|-------|
| ۲۰۲۶-۰۶-۲۳ | — | ایجاد ROADMAP.md | نقشه راه کامل نوشته شد |
| ۲۰۲۶-۰۶-۲۳ | ۰ | زیرساخت baseline | metrics, evaluate, health-check ساخته شد |
| ۲۰۲۶-۰۶-۲۳ | ۰ | اندازه‌گیری CER | CER=1.0 (۰٪) — VLM بدون vision |
| ۲۰۲۶-۰۶-۲۳ | ۱ | اسکریپت‌های فاز ۱ | filter, clean, extract_lines, split, download |
| ۲۰۲۶-۰۶-۲۳ | ۱ | فیلتر تصاویر | ۱۲۶ تصویر باقی، ۷ رد شد |
| ۲۰۲۶-۰۶-۲۳ | ۱ | OpenRouter vision | auto_labeler تست OK |
| ۲۰۲۶-۰۶-۲۳ | ۱ | extract_lines | ۵ خط از ۳ تصویر |

---

## دستورالعمل برای Agent (Cursor)

1. **قبل از هر کار** این فایل را بخوان.
2. **بعد از هر کار** تیک بزن، تاریخ بزن، توضیح کوتاه بنویس.
3. اگر CER یا آمار جدید داری، در جدول و فیلدهای `___` به‌روز کن.
4. اگر به GPU نیاز شد، فاز ۳ را متوقف کن و به کاربر بگو Colab را اجرا کند.
5. وقتی کاربر مدل آورد، در `models/trocr-fa-exam-v1/` قرار بده و فاز ۴ را شروع کن.

---

## ریسک‌ها و راه‌حل

| ریسک | احتمال | راه‌حل |
|------|--------|--------|
| برچسب‌های VLM نادرست | بالا | فیلتر هالوسینیشن + بازبینی ۱۰٪ نمونه‌ها |
| داده کم برای دست‌نویس امتحان | متوسط | Sadri/Iranshahr + crawler + augmentation |
| TrOCR فقط خط تکی می‌خواند | قطعی | Surya line detection قبل از OCR |
| API رایگان قطع شود | متوسط | TrOCR لوکال به‌عنوان primary |
| Colab timeout | متوسط | checkpoint هر ۲ epoch در Drive |

---

## تخمین زمان کل

| فاز | زمان | GPU |
|-----|------|-----|
| ۰ — زیرساخت | ۲–۳ روز | ❌ |
| ۱ — داده | ۱–۲ هفته | ❌ |
| ۲ — pipeline | ۳–۵ روز | ❌ |
| ۳ — آموزش Colab | ۱ هفته | ✅ |
| ۴ — ادغام | ۳–۵ روز | اختیاری |
| ۵ — بهبود تا ۹۵٪ | ۲–۴ هفته | ✅ (re-train) |
| **جمع** | **~۶–۱۰ هفته** | |

---

*آخرین ویرایش: ۲۰۲۶-۰۶-۲۳*
