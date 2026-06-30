#!/usr/bin/env python3
"""
ارزیابی جامع روی دیتاست داخلی: صفحات برچسب‌خورده + کراپ خط دست‌نویس + نمونه تصادفی.

Usage:
    python scripts/eval_dataset_full.py
    python scripts/eval_dataset_full.py --raw-sample 30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

from training.eval.evaluate import (  # noqa: E402
    EvalReport,
    blocks_to_text,
    evaluate_dataset,
    run_pipeline_on_image,
)
from training.eval.label_quality import is_valid_label, persian_char_ratio  # noqa: E402
from training.eval.metrics import accuracy_from_cer, cer, wer  # noqa: E402


@dataclass
class RawImageResult:
    image_path: str
    block_count: int
    text_blocks: int
    cropped_blocks: int
    illegible_blocks: int
    extracted_any_text: bool
    pipeline_ok: bool
    error: str | None = None


def evaluate_raw_sample(raw_dir: Path, *, sample_size: int, seed: int) -> list[RawImageResult]:
    images = sorted(
        p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
    )
    if not images:
        return []

    rng = random.Random(seed)
    chosen = images if len(images) <= sample_size else rng.sample(images, sample_size)

    results: list[RawImageResult] = []
    for index, path in enumerate(chosen, start=1):
        rel = str(path.relative_to(PROJECT_ROOT))
        print(f"[raw {index}/{len(chosen)}] {path.name}", flush=True)
        try:
            blocks, ocr_error = run_pipeline_on_image(path, run_ocr=True)
            text_blocks = sum(1 for b in blocks if b.get("text") and str(b["text"]).strip())
            cropped = sum(1 for b in blocks if b.get("crop_image") is not None)
            illegible = sum(1 for b in blocks if b.get("illegible"))
            hypothesis = blocks_to_text(blocks)
            results.append(
                RawImageResult(
                    image_path=rel,
                    block_count=len(blocks),
                    text_blocks=text_blocks,
                    cropped_blocks=cropped,
                    illegible_blocks=illegible,
                    extracted_any_text=bool(hypothesis.strip()),
                    pipeline_ok=ocr_error is None,
                    error=ocr_error,
                )
            )
        except Exception as exc:
            results.append(
                RawImageResult(
                    image_path=rel,
                    block_count=0,
                    text_blocks=0,
                    cropped_blocks=0,
                    illegible_blocks=0,
                    extracted_any_text=False,
                    pipeline_ok=False,
                    error=str(exc),
                )
            )
    return results


def _persian_handwriting_samples(report: EvalReport) -> list[dict]:
    """نمونه‌هایی که برچسب فارسی دارند (دست‌نویس)."""
    rows = []
    for sample in report.evaluated:
        ref = sample.reference or ""
        if persian_char_ratio(ref) >= 0.3:
            rows.append(asdict(sample))
    return rows


def _print_section(title: str, summary: dict) -> None:
    print(f"\n{'=' * 64}")
    print(title)
    print(f"{'=' * 64}")
    if not summary.get("count"):
        print("بدون نمونه معتبر برای ارزیابی دقت.")
        return
    print(f"نمونه‌های ارزیابی‌شده: {summary['count']}")
    print(f"دقت کاراکتری میانگین: {summary['mean_accuracy_pct']}%")
    print(f"CER میانگین: {summary['mean_cer']} | WER میانگین: {summary['mean_wer']}")
    print(
        f"نرخ موفقیت: "
        f"تطابق کامل {summary['exact_match']}/{summary['count']} "
        f"({100*summary['exact_match']/summary['count']:.1f}%) | "
        f"CER≤5% → {summary['cer_le_5pct']}/{summary['count']} "
        f"({100*summary['cer_le_5pct']/summary['count']:.1f}%) | "
        f"CER≤20% → {summary['cer_le_20pct']}/{summary['count']} "
        f"({100*summary['cer_le_20pct']/summary['count']:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full internal dataset evaluation")
    parser.add_argument("--raw-sample", type=int, default=25, help="Random raw pages (0=skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "logs" / "eval_full_report.json")
    args = parser.parse_args()

    print("Exam OCR — ارزیابی دیتاست داخلی")
    print(f"زمان: {datetime.now(timezone.utc).isoformat()}")

    print("\n>>> ارزیابی صفحات برچسب‌خورده (dataset.jsonl)...", flush=True)
    pages_report = evaluate_dataset(
        PROJECT_ROOT / "dataset" / "dataset.jsonl",
        run_ocr=True,
        only_valid_labels=True,
    )
    _print_section("صفحات کامل (برچسب معتبر)", pages_report.summary())

    print("\n>>> ارزیابی کراپ خط دست‌نویس (lines.jsonl)...", flush=True)
    lines_report = evaluate_dataset(
        PROJECT_ROOT / "dataset" / "lines.jsonl",
        run_ocr=True,
        only_valid_labels=True,
    )
    _print_section("کراپ خط (دست‌نویس فارسی)", lines_report.summary())

    persian_lines = _persian_handwriting_samples(lines_report)
    if persian_lines:
        persian_cers = [row["cer"] for row in persian_lines]
        mean_cer = sum(persian_cers) / len(persian_cers)
        exact = sum(1 for c in persian_cers if c == 0.0)
        le5 = sum(1 for c in persian_cers if c <= 0.05)
        print(
            f"\nفقط خطوط فارسی ({len(persian_lines)} نمونه): "
            f"دقت {accuracy_from_cer(mean_cer):.1f}% | "
            f"تطابق کامل {exact}/{len(persian_lines)} | CER≤5% {le5}/{len(persian_lines)}"
        )

    raw_results: list[RawImageResult] = []
    if args.raw_sample > 0:
        print(f"\n>>> نمونه عملیاتی از raw_images (n={args.raw_sample})...", flush=True)
        raw_results = evaluate_raw_sample(
            PROJECT_ROOT / "dataset" / "raw_images",
            sample_size=args.raw_sample,
            seed=args.seed,
        )
        ok = [r for r in raw_results if r.pipeline_ok]
        with_text = [r for r in raw_results if r.extracted_any_text]
        print(f"\n{'=' * 64}")
        print(f"نمونه تصادفی raw_images ({len(raw_results)} تصویر)")
        print(f"{'=' * 64}")
        if raw_results:
            print(
                f"پردازش موفق: {len(ok)}/{len(raw_results)} "
                f"({100*len(ok)/len(raw_results):.1f}%)"
            )
            print(
                f"حداقل یک متن استخراج شد: {len(with_text)}/{len(raw_results)} "
                f"({100*len(with_text)/len(raw_results):.1f}%)"
            )
            if ok:
                avg_text_ratio = sum(
                    r.text_blocks / max(r.block_count, 1) for r in ok
                ) / len(ok)
                print(f"میانگین نسبت بلوک متنی: {100*avg_text_ratio:.1f}%")

    # ترکیب همه نمونه‌های برچسب‌دار
    combined = pages_report.evaluated + lines_report.evaluated
    if combined:
        mean_cer = sum(s.cer for s in combined) / len(combined)
        mean_wer = sum(s.wer for s in combined) / len(combined)
        exact = sum(1 for s in combined if s.cer == 0.0)
        le5 = sum(1 for s in combined if s.cer <= 0.05)
        le20 = sum(1 for s in combined if s.cer <= 0.20)
        print(f"\n{'=' * 64}")
        print("جمع‌بندی کل (صفحات + خطوط برچسب‌دار)")
        print(f"{'=' * 64}")
        print(f"نمونه: {len(combined)}")
        print(f"دقت کاراکتری میانگین: {accuracy_from_cer(mean_cer):.2f}%")
        print(f"CER میانگین: {mean_cer:.4f} | WER میانگین: {mean_wer:.4f}")
        print(
            f"موفقیت: تطابق کامل {exact}/{len(combined)} ({100*exact/len(combined):.1f}%) | "
            f"CER≤5% {le5}/{len(combined)} ({100*le5/len(combined):.1f}%) | "
            f"CER≤20% {le20}/{len(combined)} ({100*le20/len(combined):.1f}%)"
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pages": {"summary": pages_report.summary(), "samples": [asdict(s) for s in pages_report.samples]},
        "lines": {"summary": lines_report.summary(), "samples": [asdict(s) for s in lines_report.samples]},
        "raw_sample": [asdict(r) for r in raw_results],
        "combined_labeled_summary": {
            "count": len(combined),
            "mean_cer": round(sum(s.cer for s in combined) / len(combined), 4) if combined else None,
            "mean_accuracy_pct": round(accuracy_from_cer(sum(s.cer for s in combined) / len(combined)), 2)
            if combined
            else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nگزارش ذخیره شد: {args.output}")


if __name__ == "__main__":
    main()
