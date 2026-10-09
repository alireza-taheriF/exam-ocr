# PLAN — سرویس OCR برگهٔ امتحانی با ویرایش دانشجو و یادگیری از اصلاحات

> این فایل **سند زندهٔ پروژه** است: معماری هدف، وضعیت فعلی، فازها، چک‌لیست‌ها و تصمیم‌های باز.
> هر بار که کاری انجام شد، همین‌جا تیک بزن / وضعیت را به‌روز کن.
> آخرین به‌روزرسانی: ۲۰۲۶-۱۰-۰۹

## ۰. هدف محصول

دانشجو عکس/اسکن برگهٔ امتحانی را آپلود می‌کند ← سیستم متن را می‌خواند و در **یک باکس قابل ویرایش** نشان می‌دهد ← دانشجو اصلاح و تأیید می‌کند ← اصلاحاتِ تأییدشده (پس از فیلتر کیفیت) برای **آموزش دوره‌ای** مدل استفاده می‌شود.

**هدف دقت:** `دقت کاراکتری = 1 − CER ≥ 90٪` روی مجموعهٔ مهرشدهٔ `eval/new_gold` (۳۰۰ خط).
تعریف دقیق (متن چاپی؟ دست‌نویس شلوغ؟ کلمه یا کاراکتر؟) هنوز باز است → بخش ۹.

> ⚠️ مدل **آنلاین** یاد نمی‌گیرد. ویرایش‌ها ذخیره می‌شوند، فیلتر می‌شوند، و مدل دوره‌ای روی Colab/GPU دوباره آموزش می‌بیند.

---

## ۱. وضعیت فعلی (واقعیت، نه آرزو)

| مورد | وضعیت |
|---|---|
| دقت | ≈ **۶۲٫۳٪** روی ۱۶۸ خط `dataset/gold.jsonl` (Qwen3-VL-2B). خطوط بلند ≈ ۶۸٪، خطوط ≤ ۴ حرف ≈ ۱۹٪ |
| پیکره | ۲۲٬۲۵۰ کراپ خط (`dataset/lines_v3`) **بدون برچسب**؛ split سطح سند بدون نشتی |
| LoRA v1 | روی ۵۸۶ ردیف (۶۸ gold + ۹۶ oversample + ۴۲۲ pseudo)، در step 224/294 با OOM روی MPS قطع شد؛ بهبود اثبات‌نشده |
| `eval/new_gold` | ۳۰۰ خط مهرشده، **هنوز برچسب‌زده نشده** (UI در `labeling/` آماده) |
| وب‌اپ | `main.py`: آپلود → Surya → OCR → Word. `pipeline/ocr.py` به VLM API ریموت وصل است که در دسترس نیست. `jobs` درون حافظه. UI ویرایش ندارد |
| تداخل نسخه | `requirements.txt`: `surya-ocr` نیاز به `transformers<4.46`، Qwen3-VL احتمالاً نسخهٔ جدیدتر → **UNVERIFIED**، در فاز ۰ تأیید می‌شود |

---

## ۲. معماری هدف

```
                     ┌──────────────────────────── Browser (RTL, موبایل) ───────────────────────────┐
                     │  آپلود  →  تصویر + باکس خطوط  |  باکس متن قابل ویرایش  |  تأیید / دانلود     │
                     └──────────────┬───────────────────────────────▲────────────────────────────────┘
                                    │ REST                          │ polling / SSE
                         ┌──────────▼───────────── API (FastAPI) ───┴──────────┐
                         │ auth · upload+sha256 dedup · pages/lines/corrections │
                         └───────┬───────────────────────────────┬──────────────┘
                                 │ enqueue                       │ SQL
                         ┌───────▼────────┐              ┌───────▼────────┐
                         │ Redis + Celery │              │  PostgreSQL    │
                         └───────┬────────┘              └────────────────┘
                                 │
                 ┌───────────────▼─────────────────────────────────────────┐
                 │ OCR worker (GPU/CPU)                                    │
                 │  preprocess (deskew/rotate) → Surya line detect         │
                 │  → OcrEngine.recognize(crop) → text + token logprobs    │
                 │  (Qwen3-VL-2B + LoRA فعال از model registry)            │
                 └───────────────┬─────────────────────────────────────────┘
                                 │
   ┌─────────────────────────────▼──────────────── Learning loop (offline) ───────────────────────────┐
   │ corrections → quality gate → review queue (labeling UI) → verified_lines.jsonl                    │
   │ → make_colab_bundle.zip → Colab (LoRA train, GPU) → adapter → eval on new_gold + val → promote    │
   └────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### اصول معماری
1. **recognition خط‌به‌خط** (Surya فقط تشخیص خط؛ مدل VLM فقط یک کراپ تک‌خط می‌خواند). آموزش و inference هم‌شکل‌اند.
2. **`OcrEngine` یک اینترفیس است** (`recognize(crop) -> (text, confidence, token_probs)`)؛ backend محلی Qwen/LoRA، و در آینده هر چیز دیگر.
3. **هر پیش‌بینی `model_version` دارد** تا بشود اثر هر نسخه را اندازه گرفت و rollback کرد.
4. **`new_gold` هرگز وارد هیچ مسیر آموزشی نمی‌شود** (path + SHA-256، و `make_colab_bundle.py` در صورت وجود آن را رد می‌کند).
5. **قبل از هر داده‌ٔ جدید dedup با SHA-256 روی تصاویر** (گزارش leakage تکراری‌ها را نمی‌بیند).
6. دو stack نرم‌افزاری ممکن است تداخل داشته باشند → سرویس layout و سرویس recognition قابل جداسازی طراحی می‌شوند.

### مدل داده (Postgres)
| جدول | ستون‌های کلیدی |
|---|---|
| `users` | id, email, password_hash, training_consent(bool), created_at |
| `pages` | id, user_id, sha256(unique per user), image_path, status, model_version, created_at |
| `lines` | id, page_id, idx, bbox, crop_path, model_text, confidence, token_probs(json) |
| `corrections` | id, line_id, user_id, text, confirmed(bool), edit_cer_vs_model, quality_flag, created_at |
| `training_items` | id, correction_id, weight, review_status(pending/approved/rejected), split(train/val), exported_in(batch) |
| `models` | version, base_model, adapter_path, new_gold_cer, val_cer, active(bool), created_at |

### API (پیش‌نویس)
- `POST /api/pages` آپلود (jpg/png/tiff/pdf) → `{page_id}`
- `GET /api/pages/{id}` وضعیت + خطوط `{idx,bbox,text,confidence}`
- `PUT /api/lines/{id}` ذخیرهٔ ویرایش (autosave)
- `POST /api/pages/{id}/confirm` تأیید نهایی
- `GET /api/pages/{id}/export?fmt=docx|txt`
- ادمین: `GET /admin/review`, `POST /admin/models/{v}/promote`

---

## ۳. نقشهٔ فایل‌ها (امروز → هدف)

| مسیر | نقش | وضعیت |
|---|---|---|
| `pipeline/{preprocess,layout,crop,normalizer,export}.py` | پردازش تصویر و Word | موجود |
| `pipeline/ocr.py`, `openai_compat.py` | OCR با API ریموت | **جایگزین** با `OcrEngine` محلی |
| `app/` (api, models, tasks, engines) | بک‌اند جدید | **ساخته می‌شود** (فاز ۲) |
| `static/` → frontend ویرایشگر | UI | **بازنویسی** (فاز ۲) |
| `eval/` (split, metrics, leakage, new_gold_report) | ارزیابی | موجود |
| `labeling/` | UI برچسب‌زنی/بازبینی | موجود؛ در فاز ۳ صف بازبینی تولید هم به آن وصل می‌شود |
| `training/train_qwen_vl_lora.py` | LoRA (cuda/mps) | موجود، `--max-long-side` اضافه شد |
| `scripts/make_colab_bundle.py` | بستهٔ داده برای Colab | **جدید** |
| `notebooks/colab_train.ipynb` | اجرای آموزش روی Colab | **جدید** |
| `docs/training_oom.md` | یادداشت OOM | موجود |

---

## ۴. فاز ۰ — پایه‌ها (۲–۳ روز)
- [x] ثبت پلن (همین فایل)
- [x] commit/push کارهای ثبت‌نشده (labeling, training, eval/new_gold, …)
- [x] اسکریپت bundle برای Colab + نوت‌بوک
- [x] `--max-long-side` در trainer برای رفع OOM
- [ ] **برچسب‌زنی `new_gold` توسط Ali** (≈ ۱ ساعت): `.venv/bin/python -m labeling.ui --queue new_gold`
- [ ] اجرای `.venv/bin/python -m eval.new_gold_report` → ثبت baseline رسمی در بخش ۱
- [ ] بررسی تداخل `transformers` (Surya vs Qwen3-VL) و تصمیم: یک venv یا دو سرویس
- [ ] pin کردن نسخه‌ها (`requirements-api.txt`, `requirements-train.txt`)
- [ ] پوشه‌بندی ROADMAP/REPORTها به `docs/`

## ۵. فاز ۱ — رساندن مدل به هدف دقت (۴–۸ هفته)

### ۵.۱ دادهٔ برچسب‌خورده (گلوگاه اصلی)
- [ ] کراپ‌های عریض‌تر از ~۲۵۰px را با مدل فعلی pseudo-label بزن (۴۵٪ پیکره)
- [ ] اولویت بازبینی با **confidence پایین** (active learning)، اصلاح با UI: هدف **۳–۵ هزار خط تأییدشده**
- [ ] کراپ‌های کوتاه: برچسب خالی/تصویر (مطابق رفتار طراحی‌شده)
- [ ] ترکیب با synthetic (`dataset/qwen_ft/synthetic.jsonl`) و دیتاست‌های خارجی (`scripts/download_external_datasets.py`) با وزن کمتر
- [ ] dedup SHA-256 قبل از هر ورود داده

### ۵.۲ آموزش (Colab)
روند کامل در بخش ۷. سه آزمایش به ترتیب:
1. LoRA r=16، `--max-long-side 768`، داده‌ٔ in-domain تأییدشده
2. افزایش داده + r=32 / epoch بیشتر
3. مدل بزرگ‌تر (Qwen2.5/3-VL 7B یا مشابه) فقط اگر ۱–۲ کافی نبود و GPU اجازه داد

### ۵.۳ ارزیابی و confidence
- [ ] تنها معیار تصمیم: `val` (برای انتخاب)، و `new_gold` **فقط برای گزارش نهایی هر نسخه**
- [ ] token logprob → امتیاز اطمینان هر کلمه/خط (برای هایلایت در UI)
- [ ] گزارش «پوشش در برابر دقت»: چند درصد خطوط با confidence بالا و دقتشان

**گیت خروج:** `1−CER ≥ ۰٫۹۰` روی `new_gold` (طبق تعریف نهایی بخش ۹). در غیر این صورت پلن را تعدیل می‌کنیم.

## ۶. فاز ۲ — MVP محصول (۲–۳ هفته، موازی با فاز ۱)
- [ ] `app/` با FastAPI + SQLAlchemy + Alembic؛ SQLite برای dev، Postgres برای prod
- [ ] auth ساده (ثبت‌نام/ورود) + checkbox رضایت برای آموزش
- [ ] آپلود با sha256، محدودیت حجم/نوع، پاکسازی خودکار
- [ ] Celery + Redis: pipeline = preprocess → Surya → `OcrEngine`
- [ ] `OcrEngine` محلی (Qwen3-VL + LoRA از registry)، fallback CPU
- [ ] Frontend RTL: تصویر + overlay خطوط ↔ باکس متن؛ هایلایت کلمات کم‌اطمینان؛ autosave؛ «تأیید»؛ دانلود Word/txt
- [ ] تست API (pytest) و تست E2E با ۳ تصویر نمونه
- [ ] Dockerfile + docker-compose (api, worker, redis, postgres)

## ۷. جریان کار Colab (قابل حمل)

**محلی (Mac):**
```bash
# ۱) بستهٔ داده (فقط تصاویر لازم؛ new_gold رد می‌شود)
.venv/bin/python scripts/make_colab_bundle.py \
  --manifests dataset/qwen_ft/train_mix_indomain.jsonl dataset/qwen_ft/val_gold.jsonl \
  --out outputs/colab/exam_ocr_data.zip
# ۲) فایل zip را به Google Drive در پوشهٔ MyDrive/exam-ocr/ ببر
```
**Colab:** `notebooks/colab_train.ipynb` را باز کن (Runtime → GPU) و سلول‌ها را به ترتیب اجرا کن:
mount Drive ← `git clone` + checkout شاخه ← unzip دیتا ← نصب `transformers>=4.57, peft, accelerate` (نه `requirements.txt`) ← آموزش ← کپی adapter به Drive.

**بازگشت:** adapter را از Drive در `models/<name>/` بگذار و محلی ارزیابی کن:
```bash
QWEN_LORA_PATH=models/<name> .venv/bin/python -m eval.run_baseline --splits val
.venv/bin/python -m eval.new_gold_report
```
نکات: کد با git می‌رود، داده با zip؛ روی T4 احتمالاً fp16 و batch کوچک؛ روی L4/A100 bf16. مدل پایه هر بار از HF دانلود می‌شود (کش در Drive اختیاری).

## ۸. فاز ۳ — حلقهٔ یادگیری از ویرایش‌ها (۲–۳ هفته)
**فیلتر کیفیت** (ویرایش دانشجو برچسب مطمئن نیست):
1. فقط `confirmed=true` و کاربر `training_consent=true`
2. رد ویرایش‌های با `edit_cer_vs_model > 0.7` (بازنویسی کامل/نامرتبط)
3. تأیید بدون تغییر = وزن کم‌تر (کاربر سریع تأیید می‌کند)
4. نمونه‌برداری بازبینی ادمین با `labeling/` + هم‌توافقی چند کاربر روی نمونه‌های مشترک
5. dedup تصاویر؛ صفحات کاربر **بر اساس سند** به train/val می‌روند، هرگز `new_gold`

**چرخهٔ retrain:** وقتی ≥ N (مثلاً ۲۰۰۰) خط approved → export `verified_lines.jsonl` → bundle → Colab → adapter → ارزیابی `val`+`new_gold` → فقط در صورت بهبود، `models.active` عوض می‌شود. rollback با تغییر `active`.

**شاخص زنده:** نرخ ویرایش، میانگین `edit_cer_vs_model`، سهم خطوط تأییدشده بدون تغییر، به تفکیک `model_version`.

- [ ] جدول‌های `corrections/training_items/models`
- [ ] اسکریپت `scripts/export_verified.py`
- [ ] صف بازبینی در `labeling/`
- [ ] داشبورد شاخص‌ها

## ۹. فاز ۴ — استقرار و عملیات
- [ ] سرویس inference (GPU یا CPU + صف)، health/readiness
- [ ] مانیتورینگ زمان پاسخ و خطا، لاگ ساختارمند
- [ ] backup دیتابیس و تصاویر، سیاست نگهداری داده (برگهٔ امتحان = داده شخصی)
- [ ] سیاست حریم خصوصی/رضایت‌نامه

## ۱۰. تصمیم‌های باز
1. **تعریف دقیق ۹۰٪**: کاراکتری یا کلمه‌ای؟ دست‌نویس شلوغ هم یا فقط تمیز؟ (روی gate اثر مستقیم دارد)
2. مقیاس اولیه کاربران و محل استقرار (auth، هزینه، حریم خصوصی)
3. Colab Pro/GPU کافی برای مدل بزرگ‌تر از 2B؟
4. یک venv یا دو سرویس (بعد از بررسی تداخل `transformers`)

## ۱۱. ریسک‌ها
| ریسک | کاهش |
|---|---|
| ۹۰٪ روی دست‌نویس شلوغ نرسد | گزارش دقت به تفکیک نوع؛ confidence + هایلایت؛ تعریف هدف روی بخش قابل‌دستیابی |
| برچسب‌های دانشجو نویزی | فیلتر کیفیت + بازبینی ادمین + وزن‌دهی |
| نشت داده به `new_gold` | hash check، بازبینی bundle، split بر اساس سند |
| OOM / محدودیت GPU Colab | `--max-long-side`، batch کوچک + grad-accum، checkpoint روی Drive |
| داده شخصی برگه‌ها | رضایت‌نامه صریح، حذف، دسترسی محدود |
