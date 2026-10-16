# LoRA روی Colab — گام‌به‌گام

آموزش روی مک (M2 Pro، ۱۶ گیگ) با تنظیمات کم‌حافظه ممکن است: اسکریپت حالا بعد از هر step کش MPS را
آزاد می‌کند (اجرای قبلی بدون آن در step 224 با OOM قطع شد). سنجیده‌شده ۲۰۲۶-۱۰-۱۳: ~۷ ثانیه/step با
batch 1 × accum 4، اوج حافظه ۱۲٫۷ گیگ — یعنی ~۴ ساعت برای ۸۰۰۰ خط:

```bash
HF_HUB_OFFLINE=1 .venv-qwen/bin/python training/lora/train_qwen_lora.py --out models/lora_v2_mac --device mps \
    --local dataset/external/hw_lines/train.jsonl --max-samples 8000 --batch-size 1 --grad-accum 8 \
    --max-long-side 512 --max-label-tokens 96 --replay-ratio 0 --val-max 100 --save-steps 100
```

Colab (T4) سریع‌تر است و برای اجراهای بزرگ‌تر (با replay) همچنان مسیر اصلی است.
دادهٔ دست‌خط: `scripts/build_hw_lines.py` (MR3z4 روی Hub ستون تصویر ندارد؛ فقط از bundle می‌آید).

## ۰. قانون داده
`dataset/gold.jsonl`، `dataset/test_v3.jsonl` و `eval/new_gold` هرگز وارد آموزش نمی‌شوند.
`training/sealed.py` این را در `make_bundle.py` و `train_qwen_lora.py` اجبار می‌کند (اگر یک
کراپ مهرشده در manifest باشد، اسکریپت متوقف می‌شود).

> آداپتور قدیمی `models/qwen3-vl-fa-exam-lora-v1` با ۵۸۶ ردیف از **split تست** آموزش دیده
> (۶۸ خط طلایی + ۹۶ کپی + ۴۲۲ شبه‌برچسب از جزوهٔ اصول حسابداری ۱). روی gold قابل سنجش
> نیست و نباید فعال شود.

## ۱. روی مک: بستهٔ دادهٔ محلی (اختیاری)
اگر اصلاحات دانشجو یا خطوط برچسب‌خوردهٔ val دارید:

```bash
.venv-qwen/bin/python scripts/export_corrections.py --out exports/corrections_v1
.venv-qwen/bin/python training/lora/make_bundle.py \
    --manifest exports/corrections_v1/train.jsonl \
    --out outputs/colab/exam_ocr_lora_data.zip
```

فایل zip را در Google Drive در پوشهٔ `MyDrive/exam-ocr/` بگذارید. بدون این فایل، آموزش فقط
با دو دیتاست عمومی انجام می‌شود (`MR3z4/persian-handwriting-ocr` + replay از
`Persian_Arabic_TextLine_Image_Ocr_Small`).

> `dataset/val_calib.jsonl` را **داخل bundle نگذارید** اگر قرار است با آن آستانهٔ اطمینان را
> تنظیم کنید؛ دادهٔ تنظیم و آموزش جدا بمانند.

## ۲. روی Colab
1. `training/lora/train_qwen_lora_colab.ipynb` را در Colab باز کنید (File → Upload notebook،
   یا از GitHub). Runtime → Change runtime type → **T4 GPU**.
2. سلول‌ها را به ترتیب اجرا کنید. اگر مخزن private است، در سلول clone یک توکن read-only بگذارید.
3. در لاگ سلول آموزش، خط‌های `[data] ... features=...` را نگاه کنید: نام ستون‌های دیتاست‌های
   HF از این مک (بدون اینترنت) **بررسی نشده‌اند**. اگر ستون تصویر/متن اشتباه حدس زده شد،
   `--hw-image-col` / `--hw-text-col` / `--hw-bbox-col` را بدهید.
4. اگر اتصال قطع شد، همان سلول آموزش را دوباره اجرا کنید؛ از آخرین checkpoint در Drive ادامه می‌دهد.
5. سلول آخر `MyDrive/exam-ocr/lora_v2_adapter.zip` می‌سازد. در `training_meta.json` عدد
   `val_cer_before` و `val_cer_after` را ببینید (روی val همان ترکیب داده، نه gold).

تنظیمات پیش‌فرض برای T4: batch 2 × grad-accum 8، fp16 با وزن‌های LoRA در fp32، gradient
checkpointing، طول ضلع بلند تصویر ≤ ۷۶۸. با OOM: `--batch-size 1 --grad-accum 16` یا `--max-long-side 512`.

## ۳. برگشت به مک: دروازهٔ ارزیابی
```bash
unzip ~/Downloads/lora_v2_adapter.zip -d models/
HF_HUB_OFFLINE=1 .venv-qwen/bin/python scripts/eval_gate.py \
    --label lora_v2 --adapter models/lora_v2 --register
```

`models/registry.json` نتیجه را `champion` یا `rejected` ثبت می‌کند (قانون ارتقا:
`.claude/skills/ocr-eval-gate`).

## ۴. فعال‌سازی — فقط اگر champion شد
```bash
QWEN_ADAPTER_PATH=models/lora_v2 OCR_ENGINE=qwen_local .venv-qwen/bin/uvicorn main:app --port 8000
```
اگر `rejected` شد، فعال نکنید؛ گزارش `dataset/reports/lora_v2.json` را برای تفکیک سند/طول ببینید.

## آزمون خشک روی مک (فقط بررسی کد)
```bash
HF_HUB_OFFLINE=1 .venv-qwen/bin/python training/lora/train_qwen_lora.py --out /tmp/lora_dry \
    --device cpu --local <manifest-20-rows>.jsonl --max-samples 20 --max-steps 2 \
    --batch-size 2 --grad-accum 1 --max-long-side 384 --skip-eval --replay-ratio 0
```
