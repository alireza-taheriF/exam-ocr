#!/usr/bin/env python3
"""
Evaluate Exam OCR pipeline against labeled dataset (image-level CER/WER).

Usage:
    python training/eval/evaluate.py
    python training/eval/evaluate.py --dataset dataset/dataset.jsonl --limit 5
    python training/eval/evaluate.py --no-ocr   # layout only (no VLM calls)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

from training.eval.metrics import accuracy_from_cer, cer, normalize_text, wer  # noqa: E402


@dataclass
class SampleResult:
    image_path: str
    reference: str
    hypothesis: str
    cer: float
    wer: float
    block_count: int
    text_blocks: int
    cropped_blocks: int
    illegible_blocks: int
    error: str | None = None


@dataclass
class EvalReport:
    samples: list[SampleResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def evaluated(self) -> list[SampleResult]:
        return [s for s in self.samples if s.error is None]

    def summary(self) -> dict:
        ok = self.evaluated
        if not ok:
            return {
                "count": 0,
                "mean_cer": None,
                "mean_wer": None,
                "mean_accuracy_pct": None,
                "skipped": len(self.skipped),
            }
        mean_cer = sum(s.cer for s in ok) / len(ok)
        mean_wer = sum(s.wer for s in ok) / len(ok)
        return {
            "count": len(ok),
            "mean_cer": round(mean_cer, 4),
            "mean_wer": round(mean_wer, 4),
            "mean_accuracy_pct": round(accuracy_from_cer(mean_cer), 2),
            "skipped": len(self.skipped),
            "total_text_blocks": sum(s.text_blocks for s in ok),
            "total_cropped_blocks": sum(s.cropped_blocks for s in ok),
            "total_illegible_blocks": sum(s.illegible_blocks for s in ok),
            "cer_le_5pct": sum(1 for s in ok if s.cer <= 0.05),
            "cer_le_10pct": sum(1 for s in ok if s.cer <= 0.10),
            "cer_le_20pct": sum(1 for s in ok if s.cer <= 0.20),
            "exact_match": sum(1 for s in ok if s.cer == 0.0),
        }


def load_labels(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


from training.eval.label_quality import is_valid_label  # noqa: E402


def blocks_to_text(blocks: list[dict]) -> str:
    parts: list[str] = []
    for block in blocks:
        text = block.get("text")
        if text and str(text).strip():
            parts.append(str(text).strip())
    return "\n".join(parts)


def run_pipeline_on_image(image_path: Path, *, run_ocr: bool) -> tuple[list[dict], str | None]:
    from pipeline.layout import detect_blocks
    from pipeline.ocr import run_ocr
    from pipeline.preprocess import enhance_page_rgb

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        pil_fallback = __import__("PIL.Image", fromlist=["Image"]).Image.open(image_path)
        rgb = np.array(pil_fallback.convert("RGB"))
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rgb = enhance_page_rgb(rgb)
    blocks = detect_blocks(rgb)
    if run_ocr:
        try:
            blocks = run_ocr(rgb, blocks)
        except Exception as exc:
            return blocks, str(exc)
    return blocks, None


def evaluate_dataset(
    dataset_path: Path,
    *,
    limit: int | None = None,
    run_ocr: bool = True,
    only_valid_labels: bool = True,
) -> EvalReport:
    report = EvalReport()
    rows = load_labels(dataset_path)

    for row in rows:
        if limit is not None and len(report.samples) >= limit:
            break

        rel_path = row.get("image_path", "")
        reference = row.get("text", "")
        image_path = PROJECT_ROOT / rel_path

        if only_valid_labels and not is_valid_label(reference):
            report.skipped.append(f"{rel_path} (invalid/hallucinated label)")
            continue

        if not image_path.exists():
            report.skipped.append(f"{rel_path} (file missing)")
            continue

        print(f"  evaluating {rel_path}...", flush=True)
        blocks, ocr_error = run_pipeline_on_image(image_path, run_ocr=run_ocr)
        hypothesis = blocks_to_text(blocks)

        text_blocks = sum(1 for b in blocks if b.get("text") and str(b["text"]).strip())
        cropped = sum(1 for b in blocks if b.get("crop_image") is not None)
        illegible = sum(1 for b in blocks if b.get("illegible"))

        report.samples.append(
            SampleResult(
                image_path=rel_path,
                reference=reference,
                hypothesis=hypothesis,
                cer=cer(reference, hypothesis),
                wer=wer(reference, hypothesis),
                block_count=len(blocks),
                text_blocks=text_blocks,
                cropped_blocks=cropped,
                illegible_blocks=illegible,
                error=ocr_error,
            )
        )

    return report


def print_report(report: EvalReport, *, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    summary = report.summary()
    print(f"Evaluated: {summary['count']} | Skipped: {summary['skipped']}")
    if summary["count"]:
        print(f"Mean CER:       {summary['mean_cer']}")
        print(f"Mean WER:       {summary['mean_wer']}")
        print(f"Mean accuracy:  {summary['mean_accuracy_pct']}%")
        print(
            f"Success rates:  exact={summary['exact_match']}/{summary['count']} "
            f"≤5%CER={summary['cer_le_5pct']}/{summary['count']} "
            f"≤10%CER={summary['cer_le_10pct']}/{summary['count']} "
            f"≤20%CER={summary['cer_le_20pct']}/{summary['count']}"
        )
        print(f"Text blocks:    {summary['total_text_blocks']}")
        print(f"Cropped blocks: {summary['total_cropped_blocks']}")
        print(f"Illegible:      {summary['total_illegible_blocks']}")

    for sample in report.samples:
        print(f"\n--- {sample.image_path} ---")
        if sample.error:
            print(f"  OCR error: {sample.error}")
        print(f"  blocks={sample.block_count} text={sample.text_blocks} "
              f"crop={sample.cropped_blocks} illegible={sample.illegible_blocks}")
        print(f"  CER={sample.cer:.4f} WER={sample.wer:.4f}")
        ref_preview = sample.reference[:80].replace("\n", " ")
        hyp_preview = sample.hypothesis[:80].replace("\n", " ") if sample.hypothesis else "(empty)"
        print(f"  ref: {ref_preview}...")
        print(f"  hyp: {hyp_preview}...")

    if report.skipped:
        print(f"\nSkipped ({len(report.skipped)}):")
        for item in report.skipped[:10]:
            print(f"  - {item}")
        if len(report.skipped) > 10:
            print(f"  ... and {len(report.skipped) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Exam OCR pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "dataset.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ocr", action="store_true", help="Layout only, skip VLM")
    parser.add_argument(
        "--include-bad-labels",
        action="store_true",
        help="Include hallucinated labels in evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "logs" / "eval_report.json",
    )
    args = parser.parse_args()

    report = evaluate_dataset(
        args.dataset,
        limit=args.limit,
        run_ocr=not args.no_ocr,
        only_valid_labels=not args.include_bad_labels,
    )

    mode = "full pipeline" if not args.no_ocr else "layout only"
    print_report(report, title=f"Exam OCR Evaluation ({mode})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.summary(),
        "samples": [s.__dict__ for s in report.samples],
        "skipped": report.skipped,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
