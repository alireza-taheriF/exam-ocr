"""Exam OCR FastAPI application.

Student flow: upload a scanned answer sheet → each detected line is shown as its
crop next to an editable text box → student fixes the OCR text → confirms.
Every edit is stored append-only; confirmed, trusted corrections feed the
correction memory (immediate) and the retraining export (periodic).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
_HF_CACHE = BASE_DIR / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE / "hub"))

import time

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from app.service import LOW_CONFIDENCE, current_memory, is_holdout, process_document, reviewed_rows
from app.store import Store
from learning.label_trust import assess
from pipeline.engines import get_engine
from pipeline.export import export_to_word
from pipeline.segment import segmenter_name
from training.eval.metrics import accuracy_from_cer, corpus_cer

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
STATIC_DIR = BASE_DIR / "static"
LOGS_DIR = BASE_DIR / "logs"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".pdf"}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

for d in (DATA_DIR / "uploads", DATA_DIR / "exports", LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_err = logging.FileHandler(LOGS_DIR / "errors.log", encoding="utf-8")
_err.setLevel(logging.ERROR)
logging.getLogger().addHandler(_err)

store = Store(DATA_DIR / "exam_ocr.sqlite3")

app = FastAPI(title="Exam OCR", description="OCR برگهٔ امتحان با ویرایش دانشجو و یادگیری از اصلاحات")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class EditIn(BaseModel):
    text: str = Field(max_length=5000)
    author: str | None = Field(default=None, max_length=120)
    edit_ms: int | None = Field(default=None, ge=0)


def _document_or_404(doc_id: str) -> dict[str, Any]:
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="سند پیدا نشد.")
    return doc


def _line_view(row: dict[str, Any]) -> dict[str, Any]:
    current = row["edited_text"] if row["edited_text"] is not None else (row["predicted_text"] or "")
    conf = row["confidence"]
    return {
        "id": row["id"],
        "position": row["position"],
        "kind": row["kind"],
        "block_type": row["block_type"],
        "crop_url": f"/api/lines/{row['id']}/crop",
        "predicted_text": row["predicted_text"],
        "raw_text": row["raw_text"],
        "text": current,
        "confidence": conf,
        "low_confidence": row["kind"] == "text" and (conf is None or conf < LOW_CONFIDENCE),
        "applied_rules": row["applied_rules"],
        "edit_version": row["edit_version"] or 0,
        "error": row["prediction_error"],
    }


# ── pages ───────────────────────────────────────────────────────────────────
@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/health")
def health() -> dict[str, Any]:
    info: dict[str, Any] = {"status": "ok", "segmenter": segmenter_name(),
                            "engine": os.environ.get("OCR_ENGINE", "fake")}
    return info


# ── documents ───────────────────────────────────────────────────────────────
@app.post("/api/documents", status_code=202)
async def create_document(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    author: str | None = Form(default=None),
    course: str | None = Form(default=None),
    consent: bool = Form(default=False),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="فرمت فایل مجاز نیست (jpg, png, tiff, webp, pdf).")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="فایل خالی است.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="حجم فایل بیش از ۳۰ مگابایت است.")

    doc_id = str(uuid.uuid4())
    upload_path = DATA_DIR / "uploads" / f"{doc_id}{suffix}"
    upload_path.write_bytes(content)
    course = (course or "").strip() or None
    store.create_document(doc_id, file.filename or upload_path.name, author=(author or "").strip() or None,
                          course=course, consent=consent, holdout=is_holdout(doc_id))
    background.add_task(process_document, store, doc_id, upload_path, DATA_DIR)
    return {"id": doc_id, "status": "queued"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return store.list_documents()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    doc = _document_or_404(doc_id)
    lines = [_line_view(r) for r in store.document_lines(doc_id)] if doc["status"] == "done" else []
    return {**doc, "confirmed": doc["confirmed_at"] is not None, "lines": lines}


@app.post("/api/documents/{doc_id}/confirm")
def confirm_document(doc_id: str) -> dict[str, Any]:
    doc = _document_or_404(doc_id)
    if doc["status"] != "done":
        raise HTTPException(status_code=409, detail="پردازش سند هنوز تمام نشده است.")
    if doc["confirmed_at"] is None:
        store.update_document(doc_id, confirmed_at=time.time())
    return {"id": doc_id, "confirmed": True}


@app.get("/api/documents/{doc_id}/export")
def export_document(doc_id: str, format: str = "docx"):
    doc = _document_or_404(doc_id)
    if doc["status"] != "done":
        raise HTTPException(status_code=409, detail="پردازش سند هنوز تمام نشده است.")
    lines = [_line_view(r) for r in store.document_lines(doc_id)]
    if format == "txt":
        body = "\n".join(l["text"] for l in lines if l["kind"] == "text" and l["text"].strip())
        return PlainTextResponse(body, headers={"Content-Disposition": f'attachment; filename="{doc_id}.txt"'})
    if format != "docx":
        raise HTTPException(status_code=400, detail="format باید docx یا txt باشد.")
    blocks: list[dict[str, Any]] = []
    for line, row in zip(lines, store.document_lines(doc_id)):
        if line["kind"] == "image" or not line["text"].strip():
            blocks.append({"crop_image": Image.open(row["crop_path"]).convert("RGB")})
        else:
            blocks.append({"text": line["text"]})
    out = DATA_DIR / "exports" / f"{doc_id}.docx"
    export_to_word(blocks, str(out))
    return FileResponse(out, filename=f"exam_{doc_id}.docx",
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ── lines ───────────────────────────────────────────────────────────────────
@app.get("/api/lines/{line_id}/crop")
def line_crop(line_id: int) -> FileResponse:
    line = store.get_line(line_id)
    if line is None or not Path(line["crop_path"]).exists():
        raise HTTPException(status_code=404, detail="تصویر خط پیدا نشد.")
    return FileResponse(line["crop_path"], media_type="image/png")


@app.put("/api/lines/{line_id}")
def edit_line(line_id: int, body: EditIn) -> dict[str, Any]:
    line = store.get_line(line_id)
    if line is None:
        raise HTTPException(status_code=404, detail="خط پیدا نشد.")
    if line["kind"] != "text":
        raise HTTPException(status_code=400, detail="این بخش تصویر است و متن ندارد.")
    doc = _document_or_404(line["document_id"])
    if doc["confirmed_at"] is not None:
        raise HTTPException(status_code=409, detail="سند تأیید نهایی شده و دیگر قابل ویرایش نیست.")
    text = body.text.replace("\r", "").replace("\n", " ").strip()
    result = store.add_edit(line_id, text, author=body.author or doc["author"], edit_ms=body.edit_ms)
    return {"line_id": line_id, **result}


@app.get("/api/lines/{line_id}/history")
def line_history(line_id: int) -> dict[str, Any]:
    if store.get_line(line_id) is None:
        raise HTTPException(status_code=404, detail="خط پیدا نشد.")
    return store.line_history(line_id)


# ── learning / metrics ──────────────────────────────────────────────────────
@app.get("/api/stats")
def stats() -> dict[str, Any]:
    rows = reviewed_rows(store)
    by_version: dict[str, list[dict[str, Any]]] = {}
    trust_counts: dict[str, int] = {}
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        by_version.setdefault(row["model_version"], []).append(row)
        if row["version"]:
            t = assess(row["predicted_text"], row["final_text"], row["confidence"])
            trust_counts[t.status] = trust_counts.get(t.status, 0) + 1
            if t.status != "trusted":
                quarantined.append({"line_id": row["line_id"], "document_id": row["document_id"],
                                    "predicted": row["predicted_text"], "final": row["final_text"],
                                    "status": t.status, "reasons": t.reasons})

    versions = []
    for version, vrows in sorted(by_version.items()):
        refs = [r["final_text"] for r in vrows]
        shown = corpus_cer(refs, [r["predicted_text"] for r in vrows])
        raw = corpus_cer(refs, [r["raw_text"] for r in vrows])
        versions.append({
            "model_version": version,
            "lines": len(vrows),
            "edited_lines": sum(1 for r in vrows if r["version"]),
            "post_edit_accuracy_pct": round(accuracy_from_cer(shown), 2),
            "raw_model_accuracy_pct": round(accuracy_from_cer(raw), 2),
        })

    memory = current_memory(store)
    return {
        "engine": get_engine().version,
        "note": "دقت پس‌از‌ویرایش = شباهت متن نمایش‌داده‌شده با متن تأییدشدهٔ دانشجو، فقط روی اسناد تأییدشده.",
        "versions": versions,
        "trust": trust_counts,
        "quarantined": quarantined[:200],
        "rules": [r.as_dict() for r in memory.rules],
        "candidates": [r.as_dict() for r in memory.candidates[:50]],
    }
