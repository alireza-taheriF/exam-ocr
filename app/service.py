"""Document processing: page rendering → line segmentation → OCR → correction memory → DB."""

from __future__ import annotations

import hashlib
import logging
import os
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from app.store import Store
from learning.correction_memory import Memory, apply_rules, build_memory
from pipeline.crop import crop_block
from pipeline.engines import get_engine
from pipeline.normalizer import vlm_artifact_reason
from pipeline.segment import segment_page

logger = logging.getLogger(__name__)

PDF_RENDER_SCALE = 300 / 72
LOW_CONFIDENCE = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.6"))


def is_holdout(doc_id: str) -> bool:
    """Deterministic document-level hold-out (never used for learning; see ocr-eval-gate)."""
    fraction = float(os.environ.get("HOLDOUT_FRACTION", "0.1"))
    bucket = int(hashlib.sha1(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def reviewed_rows(store: Store) -> list[dict[str, Any]]:
    return store.final_pairs(confirmed_only=True) + store.confirmed_untouched_pairs()


def current_memory(store: Store) -> Memory:
    return build_memory(reviewed_rows(store))


def render_pages(upload_path: Path, pages_dir: Path, doc_id: str) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    if upload_path.suffix.lower() != ".pdf":
        if cv2.imread(str(upload_path)) is None:
            raise ValueError("فایل تصویر قابل خواندن نیست.")
        return [upload_path]
    pdf = pdfium.PdfDocument(str(upload_path))
    if len(pdf) == 0:
        raise ValueError("PDF هیچ صفحه‌ای ندارد.")
    paths = []
    for i in range(len(pdf)):
        path = pages_dir / f"{doc_id}_p{i + 1}.png"
        pdf[i].render(scale=PDF_RENDER_SCALE).to_pil().save(path)
        paths.append(path)
    return paths


def _raw_crop(page_rgb: np.ndarray, bbox: list[int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    return Image.fromarray(page_rgb[y1:y2, x1:x2])


def process_document(store: Store, doc_id: str, upload_path: Path, data_dir: Path) -> None:
    doc = store.get_document(doc_id)
    crops_dir = data_dir / "crops" / doc_id
    crops_dir.mkdir(parents=True, exist_ok=True)
    try:
        engine = get_engine()
        memory = current_memory(store)
        rules = memory.rules_for(doc.get("course") if doc else None)
        store.update_document(doc_id, status="processing", step="rendering", model_version=engine.version)
        page_paths = render_pages(upload_path, data_dir / "pages", doc_id)

        position = 0
        for page_index, page_path in enumerate(page_paths, start=1):
            store.update_document(doc_id, step=f"page {page_index}/{len(page_paths)}")
            bgr = cv2.imread(str(page_path))
            page_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            page_id = store.add_page(doc_id, page_index, str(page_path))

            for item in segment_page(page_rgb):
                bbox = [int(v) for v in item["bbox"]]
                if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                    continue
                position += 1
                crop_path = crops_dir / f"{position:04d}.png"
                if item["kind"] == "image":
                    crop_block(page_rgb, bbox, rgb_input=True).save(crop_path)
                    store.add_line(doc_id, page_id, position, kind="image",
                                   block_type=item.get("block_type"), bbox=bbox, crop_path=str(crop_path))
                    continue

                crop = _raw_crop(page_rgb, bbox)
                crop.save(crop_path)
                line_id = store.add_line(doc_id, page_id, position, kind="text",
                                         block_type=item.get("block_type"), bbox=bbox,
                                         crop_path=str(crop_path))
                error = None
                try:
                    raw_text, confidence = engine.recognize(crop)
                except Exception as exc:
                    logger.warning("OCR failed on line %s: %s", line_id, exc)
                    raw_text, confidence, error = "", 0.0, str(exc)
                if raw_text and vlm_artifact_reason(raw_text):
                    error = f"artifact:{vlm_artifact_reason(raw_text)}"
                    raw_text = ""
                text, applied = apply_rules(raw_text, rules)
                store.add_prediction(line_id, model_version=engine.version, raw_text=raw_text,
                                     text=text, confidence=confidence, applied_rules=applied,
                                     error=error)

        store.update_document(doc_id, status="done", step=None)
    except Exception as exc:
        logger.error("processing failed for %s: %s\n%s", doc_id, exc, traceback.format_exc())
        store.update_document(doc_id, status="failed", error=str(exc))
