# پرامپت اجرای مرحلهٔ بعد روی مک (Claude Code در VS Code)

کل متن داخل بلوک زیر را در Claude Code (اکستنشن VS Code) کپی کن. پوشهٔ باز در VS Code باید
ریشهٔ همین مخزن روی مک باشد؛ همان‌جایی که `dataset/lines_v3/` و `dataset/raw_images/` وجود دارند.

---

```text
تو روی مخزن exam-ocr روی مک‌بوک من (M2 Pro، ۱۶ گیگ رم، بدون GPU انویدیا) کار می‌کنی.
تصاویر دیتاست (dataset/lines_v3/ و dataset/raw_images/) فقط روی همین مک هستند؛ نسخهٔ ابری
پروژه به آن‌ها دسترسی نداشت. پس کار تو این است که چیزهایی را که آنجا قابل‌اجرا نبود، این‌جا
واقعاً اجرا کنی و عدد واقعی بدهی.

## قبل از هر کاری
1. برنچ را بگیر:  git fetch origin && git checkout claude/plan-90-feedback-loop && git pull
2. این فایل‌ها را کامل بخوان و از آن‌ها پیروی کن:
   - CLAUDE.md
   - PLAN_90_percent_feedback_loop.md  (پلن فعال؛ آیتم‌ها را تیک بزن و عدد واقعی را ثبت کن)
   - .claude/skills/ocr-eval-gate/SKILL.md   (قانون سنجش — اجباری)
   - .claude/skills/feedback-learning/SKILL.md
   - .claude/skills/persian-text/SKILL.md
   - REPORT_dataset_and_bakeoff.md و ROADMAP_to_79_percent.md
3. قوانین غیرقابل‌مذاکره:
   - هیچ عدد دقتی بدون اجرای واقعی گزارش نکن؛ خروجی هر سنجش در dataset/reports/ ذخیره شود.
   - dataset/gold.jsonl و test_v3 هرگز در آموزش یا تنظیم آستانه استفاده نشوند. تنظیم روی val_v3.
   - هر بار فقط یک متغیر را عوض کن و جداگانه بسنج.
   - قبل از پردازش سنگین، LM Studio و برنامه‌های پرمصرف را ببند (با LM Studio باز سرعت ۲۳ برابر کندتر بود).
   - اگر چیزی شکست خورد یا نتیجه بدتر شد، صادقانه بنویس؛ عدد را قشنگ نکن.
   - بعد از هر مرحلهٔ کامل، تست‌ها را اجرا کن، commit کن و push کن روی همین برنچ.

## مرحلهٔ ۱ — محیط و اجرای واقعی دمو
1. یک venv جدا با Python 3.12 بساز:  python3.12 -m venv .venv-qwen && .venv-qwen/bin/pip install -r requirements-qwen.txt
2. تست‌ها:  .venv-qwen/bin/python -m pytest tests learning pipeline/tests training/eval/tests scripts/tests
3. موتور واقعی را روی یک کراپ امتحان کن: یک اسکریپت کوتاه که pipeline/engines.py::get_engine("qwen_local")
   را روی ۵ کراپ از gold.jsonl اجرا کند و متن، confidence و زمان هر خط را چاپ کند.
   بررسی کن روی MPS اجرا می‌شود (QWEN_DEVICE=mps) و confidence عدد معقول بین ۰ و ۱ است.
   اگر compute_transition_scores یا peft با نسخهٔ transformers نصب‌شده خطا داد، درستش کن.
4. سرور را با موتور واقعی بالا بیاور:
   OCR_ENGINE=qwen_local SEGMENTER=opencv .venv-qwen/bin/uvicorn main:app --port 8000
   و با ۳ صفحهٔ واقعی از dataset/raw_images/ (حداقل یکی دست‌نویس) کل مسیر را امتحان کن:
   بارگذاری → دیدن خطوط → ویرایش → تأیید → /admin.
   اگر Playwright نصب است با آن خودکار کن و اسکرین‌شات بگیر؛ وگرنه از من بخواه در مرورگر ببینم.
5. کیفیت تقسیم خطوط OpenCV (pipeline/segment.py) را روی ۱۰ صفحهٔ واقعی متنوع بسنج: برای هر صفحه
   تصویری با جعبه‌های رسم‌شده در dataset/reports/segmentation/ بساز و خودت نگاهشان کن. مشکلات رایج
   (خطوط چسبیده، خط شکسته به چند تکه، خط‌کشی برگه، حاشیه) را گزارش کن و اصلاح کن. اگر کیفیت با Surya
   خیلی بهتر است، یک venv دوم با surya-ocr (transformers==4.45.2) بساز و segment را به‌صورت یک
   فرایند جدا (subprocess یا یک سرویس کوچک) صدا بزن تا با Qwen در یک محیط نباشد.

## مرحلهٔ ۲ — دروازهٔ ارزیابی (فاز ۴ پلن)
1. scripts/eval_gate.py بساز (بر پایهٔ scripts/eval_local_hf_gold.py، ولی از pipeline/engines.py استفاده کند):
   - ورودی: --engine، --adapter اختیاری، --gold، --label، و پارامترهای آزمایش (prompt، پیش‌پردازش).
   - خروجی: dataset/reports/<label>.json شامل CER/WER کل، دقت، تفکیک به ازای سند، طول خط
     (≤۴، ۵–۲۰، >۲۰) و چاپی/دست‌نویس، فاصلهٔ اطمینان bootstrap ۹۵٪، و مقایسه با champion.
   - resume پس از قطع شدن (مثل اسکریپت فعلی).
   - models/registry.json را بخواند/بنویسد (champion فعلی + تاریخچهٔ نسخه‌های رد/قبول).
2. baseline را دوباره با eval_gate روی ۱۶۸ خط طلایی بگیر. باید نزدیک CER=0.377 (دقت ۶۲٫۳٪) باشد؛
   اگر نبود، علت را پیدا کن قبل از ادامه.
3. کالیبراسیون confidence: روی val_v3 (نه gold!) نمودار/جدول رابطهٔ confidence با CER را بساز و
   LOW_CONFIDENCE_THRESHOLD را طوری پیشنهاد بده که بیشتر خطوط پرخطا هایلایت شوند.
   توجه: val_v3 برچسب دستی ندارد؛ اگر لازم شد ۱۰۰ تا ۱۵۰ خط از val_v3 را با
   scripts/label_gold_interactive.py خودم برچسب می‌زنم — فقط بپرس و ابزارش را آماده کن.

## مرحلهٔ ۳ — بهبودهای سریع (فاز ۵ پلن)، هر کدام جداگانه روی gold
به ترتیب امتحان کن و برای هر کدام گزارش جداگانه بساز:
1. ۳ تا ۵ prompt مختلف (از جمله prompt فارسی).
2. max_new_tokens متناسب با عرض کراپ + repetition_penalty (۱٫۰۵ و ۱٫۱).
3. بزرگ‌نمایی کراپ‌های کم‌ارتفاع تا حداقل ۶۴ پیکسل.
4. CLAHE / کشش کنتراست روی کراپ.
5. مسیریابی کراپ‌های خیلی کوتاه به «نیاز به بررسی» به‌جای متن (اثرش روی دقت و نرخ ناخوانا را جدا گزارش کن).
فقط تغییراتی که طبق قانون promotion در ocr-eval-gate بهتر بودند را در pipeline/engines.py پیش‌فرض کن.
در پلن، جدول «برآورد مسیر دقت» را با اعداد واقعی به‌روز کن.

## مرحلهٔ ۴ — آماده‌سازی fine-tune روی Colab (فاز ۶ پلن)
روی مک آموزش نده؛ فقط آماده کن تا من روی Colab اجرا کنم:
1. training/lora/train_qwen_lora.py و یک نوت‌بوک training/lora/train_qwen_lora_colab.ipynb بساز:
   - LoRA روی mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0 با peft، مناسب T4 رایگان
     (fp16، batch کوچک + gradient accumulation، gradient checkpointing).
   - داده: MR3z4/persian-handwriting-ocr (دست‌نویس واقعی) + بخشی از
     mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Small + خروجی scripts/export_corrections.py
     (train.jsonl) اگر موجود بود. نسبت replay قابل‌تنظیم.
   - همان prompt زمان inference در pipeline/engines.py استفاده شود.
   - خروجی: پوشهٔ آداپتور + training_meta.json (داده، hash، تنظیمات، loss).
   - در پایان، ارزیابی روی val و یک دستور آماده برای اجرای eval_gate روی gold پس از دانلود آداپتور.
2. یک اسکریپت بسته‌بندی بساز که کراپ‌های لازم را zip کند تا در Google Drive آپلود کنم.
3. README کوتاه training/lora/README.md با گام‌به‌گام: آپلود، اجرا، دانلود آداپتور به models/،
   اجرای eval_gate، و فعال‌سازی با QWEN_ADAPTER_PATH فقط اگر از دروازه رد شد.
4. با ۲۰ نمونه یک dry-run روی CPU مک اجرا کن (فقط ۲ قدم آموزش) تا مطمئن شوی کد خطا ندارد.

## در پایان
- PLAN_90_percent_feedback_loop.md را به‌روز کن (تیک‌ها + اعداد واقعی + مشکلات پیدا‌شده).
- یک خلاصهٔ کوتاه فارسی به من بده: چه چیزی اجرا شد، عدد واقعی دقت قبل و بعد هر تغییر،
  چه چیزی شکست خورد، و دقیقاً چه کاری را باید خودم روی Colab انجام دهم.
- هر جا تصمیمی بود که به من مربوط است (مثلاً برچسب‌زدن دستی یا انتخاب بین دو مسیر)، بپرس؛ حدس نزن.
```
