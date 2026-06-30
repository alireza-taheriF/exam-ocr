#!/usr/bin/env python3
"""
Convert scanned PDF notebooks into page images for auto_labeler.

Put PDF files here:
    dataset/raw_pdfs/your-notebook.pdf

Then run:
    python scripts/ingest_pdfs.py

Output pages go to:
    dataset/raw_images/{pdf_name}_p001.png, _p002.png, ...

Each page is then labeled by auto_labeler via OpenAI vision API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "dataset" / "raw_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw_images"
MANIFEST = PROJECT_ROOT / "dataset" / "manifest.jsonl"
INGEST_LOG = PROJECT_ROOT / "dataset" / "pdf_ingest.jsonl"

PDF_RENDER_SCALE = 300 / 72  # 300 DPI


def _slug(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe.strip("_")[:80] or "document"


def ingest_pdf(pdf_path: Path, *, output_dir: Path, dpi_scale: float) -> list[dict]:
    stem = _slug(pdf_path.stem)
    doc = pdfium.PdfDocument(str(pdf_path))
    if len(doc) == 0:
        raise ValueError(f"PDF has no pages: {pdf_path.name}")

    records: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(len(doc)):
        page_num = page_index + 1
        out_name = f"{stem}_p{page_num:03d}.png"
        out_path = output_dir / out_name

        if out_path.exists():
            records.append({
                "pdf": pdf_path.name,
                "page": page_num,
                "image_path": f"dataset/raw_images/{out_name}",
                "status": "skipped_exists",
            })
            continue

        page = doc[page_index]
        bitmap = page.render(scale=dpi_scale)
        bitmap.to_pil().save(out_path)

        records.append({
            "pdf": pdf_path.name,
            "page": page_num,
            "image_path": f"dataset/raw_images/{out_name}",
            "width": out_path.stat().st_size,
            "status": "created",
        })

        manifest_row = {
            "path": f"dataset/raw_images/{out_name}",
            "source_url": f"file://{pdf_path.name}#page={page_num}",
            "keyword": "pdf_ingest",
            "engine": "local",
            "source_pdf": pdf_path.name,
            "page": page_num,
        }
        with MANIFEST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into dataset/raw_images")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI (default 300)")
    args = parser.parse_args()

    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files in {args.pdf_dir}")
        print(f"Copy your scanned notebook PDF there, then run again.")
        return 0

    scale = args.dpi / 72
    total_pages = 0

    for pdf_path in pdfs:
        print(f"Ingesting: {pdf_path.name}")
        try:
            records = ingest_pdf(pdf_path, output_dir=args.output_dir, dpi_scale=scale)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        created = sum(1 for r in records if r["status"] == "created")
        skipped = sum(1 for r in records if r["status"] == "skipped_exists")
        total_pages += len(records)
        print(f"  pages={len(records)} created={created} skipped={skipped}")

        with INGEST_LOG.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nDone. {total_pages} pages from {len(pdfs)} PDF(s).")
    print("Next: python auto_labeler.py --resume --sleep 2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
