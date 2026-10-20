#!/usr/bin/env python3
"""
ارزیابی بدون نیاز به لیبل‌گذاری کل دیتاست.

۱) دقت (CER) فقط روی برچسب‌های موجود (lines.jsonl — ۵ خط دست‌نویس)
۲) نرخ موفقیت عملیاتی روی نمونه تصادفی raw_images (بدون ground truth)

Needs OCR_ALLOW_REMOTE=1 (it calls the remote VLM; images leave this machine).

Usage:
    OCR_ALLOW_REMOTE=1 python scripts/eval_without_labels.py
    python scripts/eval_without_labels.py --raw-sample 25 --delay 4
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

from pipeline.layout import detect_blocks  # noqa: E402
from pipeline.ocr import _prepare_ocr_region, extract_text_with_vlm, run_ocr  # noqa: E402
from pipeline.preprocess import enhance_page_rgb  # noqa: E402
from training.eval.metrics import accuracy_from_cer, cer  # noqa: E402


@dataclass
class LineResult:
    image: str
    reference: str
    hypothesis: str
    cer: float
    ok: bool


@dataclass
class RawResult:
    image: str
    block_count: int
    ocr_attempted: int
    text_blocks: int
    cropped_blocks: int
    illegible_blocks: int
    has_persian_text: bool
    extracted_any_text: bool
    error: str | None = None


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is not None:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.array(Image.open(path).convert("RGB"))


def _has_persian(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _direct_line_ocr(path: Path) -> str:
    """یک درخواست VLM روی کل کراپ خط — مناسب دست‌نویس."""
    img = _prepare_ocr_region(Image.open(path).convert("RGB"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return extract_text_with_vlm(b64)


def eval_labeled_lines(*, delay_sec: float) -> list[LineResult]:
    lines_path = PROJECT_ROOT / "dataset" / "lines.jsonl"
    if not lines_path.exists():
        return []

    rows = [json.loads(line) for line in lines_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results: list[LineResult] = []

    print(f"\n── دقت دست‌نویس ({len(rows)} خط برچسب‌خورده) ──")
    for index, row in enumerate(rows):
        if index > 0 and delay_sec > 0:
            time.sleep(delay_sec)
        rel = row["image_path"]
        path = PROJECT_ROOT / rel
        ref = row.get("text", "")
        print(f"  [{index + 1}/{len(rows)}] {path.name} ...", flush=True)
        try:
            hyp = _direct_line_ocr(path)
            c = cer(ref, hyp)
            ok = c <= 0.20
            print(f"       ref={ref!r}  hyp={hyp!r}  دقت={accuracy_from_cer(c):.0f}%")
            results.append(LineResult(image=rel, reference=ref, hypothesis=hyp, cer=c, ok=ok))
        except Exception as exc:
            print(f"       خطا: {exc}")
            results.append(LineResult(image=rel, reference=ref, hypothesis="", cer=1.0, ok=False))

    if results:
        mean_cer = sum(r.cer for r in results) / len(results)
        good = sum(1 for r in results if r.ok)
        exact = sum(1 for r in results if r.cer == 0.0)
        print(
            f"\n  جمع خطوط: دقت میانگین {accuracy_from_cer(mean_cer):.1f}% | "
            f"قابل‌قبول (CER≤20%) {good}/{len(results)} | تطابق کامل {exact}/{len(results)}"
        )
    return results


def eval_raw_sample(*, sample_size: int, seed: int, delay_sec: float, max_blocks: int) -> list[RawResult]:
    raw_dir = PROJECT_ROOT / "dataset" / "raw_images"
    images = sorted(
        p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    if not images:
        return []

    rng = random.Random(seed)
    chosen = images if len(images) <= sample_size else rng.sample(images, sample_size)

    print(f"\n── نرخ موفقیت عملیاتی ({len(chosen)} تصویر از {len(images)} — بدون لیبل) ──")
    results: list[RawResult] = []

    for index, path in enumerate(chosen, start=1):
        rel = str(path.relative_to(PROJECT_ROOT))
        print(f"  [{index}/{len(chosen)}] {path.name} ...", flush=True)
        try:
            rgb = enhance_page_rgb(_load_rgb(path))
            blocks = detect_blocks(rgb)

            # برای ارزیابی سریع‌تر، سقف تعداد بلوک OCR
            ocr_candidates = [b for b in blocks if b.get("type") in ("text", "list", "figure")][:max_blocks]
            if len(ocr_candidates) < len(blocks):
                blocks = ocr_candidates + [b for b in blocks if b not in ocr_candidates]

            if delay_sec > 0:
                time.sleep(delay_sec)
            blocks = run_ocr(rgb, blocks[:max_blocks] if max_blocks else blocks)

            texts = [str(b["text"]).strip() for b in blocks if b.get("text") and str(b["text"]).strip()]
            hypothesis = "\n".join(texts)
            text_n = len(texts)
            crop_n = sum(1 for b in blocks if b.get("crop_image") is not None)
            illeg_n = sum(1 for b in blocks if b.get("illegible"))
            ocr_attempted = sum(
                1 for b in blocks if b.get("type") in ("text", "list", "figure") and not b.get("crop_image")
            ) + illeg_n

            results.append(
                RawResult(
                    image=rel,
                    block_count=len(blocks),
                    ocr_attempted=ocr_attempted,
                    text_blocks=text_n,
                    cropped_blocks=crop_n,
                    illegible_blocks=illeg_n,
                    has_persian_text=_has_persian(hypothesis),
                    extracted_any_text=bool(hypothesis),
                )
            )
            status = "متن دارد" if hypothesis else "بدون متن"
            print(f"       بلوک={len(blocks)} متن={text_n} کراپ={crop_n} | {status}")
        except Exception as exc:
            print(f"       خطا: {exc}")
            results.append(
                RawResult(
                    image=rel,
                    block_count=0,
                    ocr_attempted=0,
                    text_blocks=0,
                    cropped_blocks=0,
                    illegible_blocks=0,
                    has_persian_text=False,
                    extracted_any_text=False,
                    error=str(exc),
                )
            )

    if results:
        ok = [r for r in results if r.error is None]
        with_text = [r for r in ok if r.extracted_any_text]
        with_fa = [r for r in ok if r.has_persian_text]
        print(f"\n  پردازش بدون خطا: {len(ok)}/{len(results)} ({100*len(ok)/len(results):.0f}%)")
        if ok:
            print(
                f"  حداقل یک متن استخراج شد: {len(with_text)}/{len(ok)} "
                f"({100*len(with_text)/len(ok):.0f}%)"
            )
            print(
                f"  متن فارسی تشخیص داده شد: {len(with_fa)}/{len(ok)} "
                f"({100*len(with_fa)/len(ok):.0f}%)"
            )
            avg_text_ratio = sum(r.text_blocks / max(r.block_count, 1) for r in ok) / len(ok)
            print(f"  میانگین نسبت بلوک‌های متنی: {100*avg_text_ratio:.0f}%")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate without full dataset labeling")
    parser.add_argument("--raw-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=4.0, help="Seconds between VLM calls")
    parser.add_argument("--max-blocks", type=int, default=12, help="Max OCR blocks per raw image")
    parser.add_argument("--skip-lines", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "logs" / "eval_no_labels.json")
    args = parser.parse_args()

    os.environ["OCR_REQUEST_DELAY_SEC"] = str(args.delay)

    print("Exam OCR — ارزیابی بدون لیبل‌گذاری کامل")
    print(f"زمان: {datetime.now(timezone.utc).isoformat()}")
    print(f"تأخیر بین درخواست‌ها: {args.delay}s")

    line_results = [] if args.skip_lines else eval_labeled_lines(delay_sec=args.delay)
    raw_results = [] if args.skip_raw else eval_raw_sample(
        sample_size=args.raw_sample,
        seed=args.seed,
        delay_sec=0,
        max_blocks=args.max_blocks,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labeled_lines": [asdict(r) for r in line_results],
        "raw_operational": [asdict(r) for r in raw_results],
        "summary": {
            "lines_count": len(line_results),
            "lines_mean_accuracy_pct": round(
                accuracy_from_cer(sum(r.cer for r in line_results) / len(line_results)), 2
            )
            if line_results
            else None,
            "lines_acceptable_pct": round(100 * sum(1 for r in line_results if r.ok) / len(line_results), 1)
            if line_results
            else None,
            "raw_sample_count": len(raw_results),
            "raw_extraction_rate_pct": round(
                100
                * sum(1 for r in raw_results if r.extracted_any_text and not r.error)
                / max(len([r for r in raw_results if not r.error]), 1),
                1,
            )
            if raw_results
            else None,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nگزارش: {args.output}")


if __name__ == "__main__":
    main()
