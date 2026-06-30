"""Exam OCR FastAPI application."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Configure HuggingFace cache and environment before pipeline imports load models.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
_HF_CACHE = BASE_DIR / ".cache" / "huggingface"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

import cv2
import numpy as np
import pypdfium2 as pdfium
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline.export import export_to_word
from pipeline.layout import detect_blocks
from pipeline.ocr import get_ocr_status, run_ocr
from pipeline.preprocess import enhance_page_rgb

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
STATIC_DIR = BASE_DIR / "static"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pdf"}
PDF_RENDER_SCALE = 300 / 72

UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

error_logger = logging.getLogger("exam_ocr.errors")
error_logger.setLevel(logging.ERROR)
if not error_logger.handlers:
    file_handler = logging.FileHandler(LOGS_DIR / "errors.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    error_logger.addHandler(file_handler)

app = FastAPI(title="Exam OCR", description="Persian exam paper OCR pipeline")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job registry: job_id -> metadata
jobs: dict[str, dict[str, Any]] = {}


def _log_error(message: str, exc: Exception | None = None) -> None:
    if exc is not None:
        error_logger.error("%s: %s", message, exc, exc_info=True)
    else:
        error_logger.error(message)


def _validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Allowed: jpg, png, tiff, pdf.",
        )
    return suffix


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is not None:
        return image
    raise ValueError(f"Could not load image: {path}")


def _pdf_to_image_paths(pdf_path: Path, job_id: str) -> list[Path]:
    """Render each PDF page to a PNG file and return image paths."""
    image_paths: list[Path] = []
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {pdf_path.name}") from exc

    if len(pdf) == 0:
        raise ValueError("PDF has no pages.")

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        bitmap = page.render(scale=PDF_RENDER_SCALE)
        pil_image = bitmap.to_pil()
        page_path = UPLOADS_DIR / f"{job_id}_page_{page_index + 1}.png"
        pil_image.save(page_path)
        image_paths.append(page_path)

    return image_paths


def _resolve_image_paths(upload_path: Path, job_id: str) -> list[Path]:
    """Return one or more image paths ready for the pipeline."""
    if upload_path.suffix.lower() == ".pdf":
        return _pdf_to_image_paths(upload_path, job_id)

    try:
        _load_image(upload_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [upload_path]


def _update_job(job_id: str, **fields: Any) -> None:
    jobs.setdefault(job_id, {"job_id": job_id})
    jobs[job_id].update(fields)


def _count_block_stats(blocks: list[dict[str, Any]]) -> tuple[int, int, int]:
    block_count = len(blocks)
    text_blocks = sum(
        1 for block in blocks if block.get("text") and str(block["text"]).strip()
    )
    cropped_blocks = sum(1 for block in blocks if block.get("crop_image") is not None)
    return block_count, text_blocks, cropped_blocks


def _ocr_failed_completely(blocks: list[dict[str, Any]]) -> bool:
    """True when no text and no cropped fallback blocks were produced."""
    if not blocks:
        return True
    has_text = any(str(block.get("text") or "").strip() for block in blocks)
    has_crop = any(block.get("crop_image") is not None for block in blocks)
    return not has_text and not has_crop


def _blocks_to_full_text(blocks: list[dict[str, Any]]) -> str:
    """Join OCR text blocks in reading order."""
    parts: list[str] = []
    for block in blocks:
        if block.get("illegible"):
            continue
        text = block.get("text")
        if text and str(text).strip():
            parts.append(str(text).strip())
    return "\n\n".join(parts)


def _blocks_to_preview(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serializable block summary for the API/UI."""
    preview: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        entry: dict[str, Any] = {
            "index": index,
            "type": block.get("type"),
            "has_text": bool(block.get("text") and str(block["text"]).strip()),
            "illegible": bool(block.get("illegible")),
            "has_crop": block.get("crop_image") is not None,
        }
        if entry["has_text"]:
            entry["text"] = str(block["text"]).strip()
        preview.append(entry)
    return preview


def _run_pipeline_for_paths(image_paths: list[Path], job_id: str) -> list[dict[str, Any]]:
    all_blocks: list[dict[str, Any]] = []

    for index, image_path in enumerate(image_paths, start=1):
        _update_job(job_id, status="processing", step=f"loading_page_{index}")

        original_bgr = _load_image(image_path)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        _update_job(job_id, step=f"preprocessing_page_{index}")
        page_rgb = enhance_page_rgb(original_rgb)

        _update_job(job_id, step=f"detecting_layout_page_{index}")
        blocks = detect_blocks(page_rgb)
        print(f"[page {index}] layout blocks detected: {len(blocks)}")

        _update_job(job_id, step=f"running_ocr_page_{index}")
        try:
            blocks = run_ocr(page_rgb, blocks)
        except Exception as exc:
            _log_error(f"OCR step failed for job {job_id}", exc)
            raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

        page_text = sum(1 for b in blocks if b.get("text") and str(b["text"]).strip())
        page_crop = sum(1 for b in blocks if b.get("crop_image") is not None)
        print(
            f"[page {index}] OCR complete: "
            f"{page_text} text blocks, {page_crop} cropped blocks"
        )

        all_blocks.extend(blocks)

    if _ocr_failed_completely(all_blocks):
        message = (
            "Pipeline produced no usable output. Check the uploaded image quality, "
            "layout detection, and VLM_API_KEY in .env."
        )
        _log_error(f"Job {job_id}: {message}")
        raise HTTPException(status_code=500, detail=message)

    total_text, total_crop = _count_block_stats(all_blocks)[1:]
    print(
        f"[job {job_id}] total: {len(all_blocks)} blocks, "
        f"{total_text} with text, {total_crop} cropped"
    )

    return all_blocks


@app.get("/")
def index_page() -> FileResponse:
    """Serve the Persian upload UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "ocr": get_ocr_status()}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload an exam image/PDF, run the full OCR pipeline, and export DOCX.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    try:
        suffix = _validate_extension(file.filename)
    except HTTPException:
        _log_error(f"Invalid upload extension: {file.filename}")
        raise

    job_id = str(uuid.uuid4())
    upload_path = UPLOADS_DIR / f"{job_id}{suffix}"
    output_file = OUTPUTS_DIR / f"result_{job_id}.docx"

    _update_job(
        job_id,
        status="processing",
        step="uploading",
        output_file=str(output_file.relative_to(BASE_DIR)),
        block_count=0,
        text_blocks=0,
        cropped_blocks=0,
        error=None,
    )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        upload_path.write_bytes(content)

        image_paths = _resolve_image_paths(upload_path, job_id)
        all_blocks = _run_pipeline_for_paths(image_paths, job_id)

        _update_job(job_id, step="exporting")
        export_to_word(all_blocks, str(output_file))

        block_count, text_blocks, cropped_blocks = _count_block_stats(all_blocks)
        full_text = _blocks_to_full_text(all_blocks)
        result = {
            "job_id": job_id,
            "status": "done",
            "output_file": str(output_file.relative_to(BASE_DIR)),
            "block_count": block_count,
            "text_blocks": text_blocks,
            "cropped_blocks": cropped_blocks,
            "full_text": full_text,
            "blocks": _blocks_to_preview(all_blocks),
        }
        _update_job(job_id, status="done", error=None, **{
            k: v for k, v in result.items() if k not in {"status", "job_id"}
        })
        return result

    except HTTPException as exc:
        _update_job(job_id, status="failed", error=str(exc.detail))
        raise
    except Exception as exc:
        _log_error(f"Unexpected pipeline failure for job {job_id}", exc)
        _update_job(job_id, status="failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc


@app.get("/download/{job_id}")
def download(job_id: str) -> FileResponse:
    """Download the generated Word document for a completed job."""
    output_path = OUTPUTS_DIR / f"result_{job_id}.docx"
    if not output_path.exists():
        _log_error(f"Download requested for missing job output: {job_id}")
        raise HTTPException(status_code=404, detail="Output file not found for this job.")

    return FileResponse(
        path=str(output_path),
        filename=f"exam_ocr_{job_id}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.get("/status/{job_id}")
def status(job_id: str) -> dict[str, Any]:
    """Return processing status and summary for a job."""
    job = jobs.get(job_id)
    if job is None:
        output_path = OUTPUTS_DIR / f"result_{job_id}.docx"
        if output_path.exists():
            return {
                "job_id": job_id,
                "status": "done",
                "output_file": str(output_path.relative_to(BASE_DIR)),
            }
        raise HTTPException(status_code=404, detail="Job not found.")

    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "step": job.get("step"),
        "output_file": job.get("output_file"),
        "block_count": job.get("block_count"),
        "text_blocks": job.get("text_blocks"),
        "cropped_blocks": job.get("cropped_blocks"),
        "full_text": job.get("full_text"),
        "blocks": job.get("blocks"),
        "error": job.get("error"),
    }
