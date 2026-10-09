#!/usr/bin/env python3
"""Local FastAPI labeling UI for sealed new_gold (bind 127.0.0.1 only).

Usage:
  python -m labeling.ui --queue new_gold
  # → http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from labeling.db import (
    DEFAULT_DB,
    DEFAULT_JSONL,
    append_correction,
    connect,
    get_model_preds,
    latest_non_skip,
    load_queue_meta,
    rebuild_jsonl,
    save_queue_meta,
    undo_last,
)
from labeling.form_guess import guess_region_type
from labeling.queue import (
    DEFAULT_BLIND_COUNT,
    DEFAULT_PREFILL_MODEL,
    DEFAULT_REPEAT_RATE,
    DEFAULT_SEED,
    build_queue_order,
    context_neighbors,
    load_manifest,
)
from labeling.validate import validate_correction

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
NEW_GOLD_DIR = PROJECT_ROOT / "eval" / "new_gold"


class SaveBody(BaseModel):
    presentation_index: int
    line_id: str
    corrected_text: str = ""
    action: str = "edited"
    region_type: str = "body"
    reviewer: str = "local"
    started_at: str | None = None
    seconds_spent: float | None = None
    notes: str = ""
    force: bool = False


def _ensure_order(conn, queue: str, *, seed: int, blind_count: int, repeat_rate: float, prefill_model: str) -> list[dict]:
    meta = load_queue_meta(conn, queue)
    if meta and meta.get("order"):
        return meta["order"]
    manifest = load_manifest()
    items = build_queue_order(
        manifest,
        seed=seed,
        blind_count=blind_count,
        repeat_rate=repeat_rate,
        prefill_model=prefill_model,
    )
    order = [it.to_dict() for it in items]
    save_queue_meta(
        conn,
        queue=queue,
        seed=seed,
        blind_count=blind_count,
        repeat_rate=repeat_rate,
        order=order,
    )
    return order


def _done_presentation_indices(conn, queue: str, order: list[dict]) -> set[int]:
    """Presentations already saved (latest non-undo action per presentation_index)."""
    rows = conn.execute(
        """
        SELECT presentation_index, action, notes, id FROM corrections
        WHERE queue=? AND presentation_index IS NOT NULL
        ORDER BY id ASC
        """,
        (queue,),
    ).fetchall()
    # Walk chronologically; undo sentinel clears that presentation
    state: dict[int, str] = {}
    for r in rows:
        idx = r["presentation_index"]
        if idx is None:
            continue
        if (r["notes"] or "") == "__UNDO__" or r["action"] == "skipped" and (r["notes"] or "") == "__UNDO__":
            state.pop(idx, None)
            continue
        if r["action"] == "skipped" and (r["notes"] or "") == "__UNDO__":
            state.pop(idx, None)
            continue
        state[int(idx)] = r["action"]
    # Also treat undo: if last row for a presentation is __UNDO__, not done
    # Recompute with last-row-wins including undo
    last_by_idx: dict[int, Any] = {}
    for r in rows:
        if r["presentation_index"] is None:
            continue
        last_by_idx[int(r["presentation_index"])] = r
    done: set[int] = set()
    for idx, r in last_by_idx.items():
        if (r["notes"] or "") == "__UNDO__":
            continue
        done.add(idx)
    return done


def create_app(
    *,
    queue: str = "new_gold",
    db_path: Path = DEFAULT_DB,
    seed: int = DEFAULT_SEED,
    blind_count: int = DEFAULT_BLIND_COUNT,
    repeat_rate: float = DEFAULT_REPEAT_RATE,
    prefill_model: str = DEFAULT_PREFILL_MODEL,
) -> FastAPI:
    app = FastAPI(title="Exam OCR Labeling (local)", docs_url="/docs")
    app.state.queue = queue
    app.state.db_path = Path(db_path)
    app.state.seed = seed
    app.state.blind_count = blind_count
    app.state.repeat_rate = repeat_rate
    app.state.prefill_model = prefill_model
    app.state.t0 = time.time()

    static_dir = STATIC_DIR
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    # Serve crop images from project root (read-only)
    if (PROJECT_ROOT / "dataset").exists():
        app.mount(
            "/dataset",
            StaticFiles(directory=str(PROJECT_ROOT / "dataset")),
            name="dataset",
        )

    def _conn():
        return connect(app.state.db_path)

    @app.get("/")
    def index():
        return FileResponse(static_dir / "index.html")

    @app.get("/api/config")
    def api_config():
        return {
            "queue": app.state.queue,
            "seed": app.state.seed,
            "blind_count": app.state.blind_count,
            "repeat_rate": app.state.repeat_rate,
            "prefill_model": app.state.prefill_model,
            "bind": "127.0.0.1",
        }

    @app.get("/api/progress")
    def api_progress():
        conn = _conn()
        order = _ensure_order(
            conn,
            app.state.queue,
            seed=app.state.seed,
            blind_count=app.state.blind_count,
            repeat_rate=app.state.repeat_rate,
            prefill_model=app.state.prefill_model,
        )
        done_idx = _done_presentation_indices(conn, app.state.queue, order)
        latest = latest_non_skip(conn, app.state.queue)
        # Timing from history
        rows = list(
            conn.execute(
                """
                SELECT seconds_spent, saved_at FROM corrections
                WHERE queue=? AND notes != '__UNDO__' AND action != 'skipped'
                """,
                (app.state.queue,),
            )
        )
        # include skipped that aren't undo? skip for rate
        spent = [float(r["seconds_spent"] or 0) for r in rows if r["seconds_spent"]]
        n_done = len(done_idx)
        n_total = len(order)
        unique_done = len(latest)
        elapsed_h = (sum(spent) / 3600.0) if spent else 0.0
        items_per_hour = (len(spent) / elapsed_h) if elapsed_h > 0 else 0.0
        return {
            "queue": app.state.queue,
            "n_total_presentations": n_total,
            "n_done_presentations": n_done,
            "n_unique_lines": len({o["line_id"] for o in order}),
            "n_unique_done": unique_done,
            "pct": round(100.0 * n_done / max(n_total, 1), 2),
            "items_per_hour": round(items_per_hour, 1),
            "blind_count": app.state.blind_count,
            "next_index": next((i for i in range(n_total) if i not in done_idx), None),
        }

    @app.get("/api/next")
    def api_next(index: int | None = Query(default=None)):
        conn = _conn()
        order = _ensure_order(
            conn,
            app.state.queue,
            seed=app.state.seed,
            blind_count=app.state.blind_count,
            repeat_rate=app.state.repeat_rate,
            prefill_model=app.state.prefill_model,
        )
        done_idx = _done_presentation_indices(conn, app.state.queue, order)
        if index is None:
            index = next((i for i in range(len(order)) if i not in done_idx), None)
            if index is None:
                return {"done": True, "progress": api_progress()}
        if index < 0 or index >= len(order):
            raise HTTPException(404, "index out of range")
        item = order[index]
        preds = get_model_preds(conn, item["line_id"])
        prefill_model = item.get("prefill_model") or app.state.prefill_model
        model_text = preds.get(prefill_model, "")
        # Also expose view-A sibling if present
        alt = {k: v for k, v in preds.items() if k != prefill_model}
        guessed = guess_region_type(model_text or "")
        ctx = context_neighbors(order, index, window=1)
        # Do NOT tell the client this is a repeat
        payload = {
            "done": False,
            "presentation_index": index,
            "line_id": item["line_id"],
            "crop_path": item["crop_path"],
            "image_url": "/" + item["crop_path"].lstrip("/"),
            "source_document": item.get("source_document") or "",
            "blind_mode": bool(item.get("blind_mode")),
            "prefill_model": prefill_model,
            "model_text": "" if item.get("blind_mode") else model_text,
            "model_text_available": bool(model_text),
            "other_models": alt if not item.get("blind_mode") else {},
            "region_type_guess": guessed,
            "context": ctx,
            "already_saved": index in done_idx,
            "progress": api_progress(),
        }
        return payload

    @app.post("/api/save")
    def api_save(body: SaveBody):
        conn = _conn()
        order = _ensure_order(
            conn,
            app.state.queue,
            seed=app.state.seed,
            blind_count=app.state.blind_count,
            repeat_rate=app.state.repeat_rate,
            prefill_model=app.state.prefill_model,
        )
        if body.presentation_index < 0 or body.presentation_index >= len(order):
            raise HTTPException(400, "bad presentation_index")
        item = order[body.presentation_index]
        if item["line_id"] != body.line_id:
            raise HTTPException(400, "line_id mismatch")
        v = validate_correction(body.corrected_text, action=body.action)
        if not v["ok"] and not body.force:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "warnings": v["warnings"]},
            )
        preds = get_model_preds(conn, body.line_id)
        prefill_model = item.get("prefill_model") or app.state.prefill_model
        model_text = "" if item.get("blind_mode") else preds.get(prefill_model, "")
        row_id = append_correction(
            conn,
            line_id=body.line_id,
            queue=app.state.queue,
            model_text=model_text,
            corrected_text=body.corrected_text,
            action=body.action,
            region_type=body.region_type if body.region_type in ("body", "form_field") else "body",
            reviewer=body.reviewer,
            started_at=body.started_at,
            seconds_spent=body.seconds_spent,
            notes=body.notes,
            blind_mode=bool(item.get("blind_mode")),
            presentation_index=body.presentation_index,
            is_repeat=bool(item.get("is_repeat")),
            prefill_model=prefill_model,
        )
        _sync_corrections_csv(conn, app.state.queue)
        return {"ok": True, "id": row_id, "warnings": v["warnings"], "progress": api_progress()}

    @app.post("/api/undo")
    def api_undo():
        conn = _conn()
        result = undo_last(conn, app.state.queue)
        if not result:
            raise HTTPException(404, "nothing to undo")
        _sync_corrections_csv(conn, app.state.queue)
        return {"ok": True, **result, "progress": api_progress()}

    @app.get("/api/export")
    def api_export():
        conn = _conn()
        path = rebuild_jsonl(conn, app.state.queue, DEFAULT_JSONL)
        csv_path = _sync_corrections_csv(conn, app.state.queue)
        return {
            "jsonl": str(path.relative_to(PROJECT_ROOT)),
            "csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "n_history": path.stat().st_size,
        }

    @app.get("/api/guess_region")
    def api_guess_region(text: str = ""):
        return {"region_type": guess_region_type(text)}

    return app


def _sync_corrections_csv(conn, queue: str) -> Path:
    """Write latest non-skip corrected_text into eval/new_gold/corrections.csv."""
    latest = latest_non_skip(conn, queue)
    manifest = {r["line_id"]: r for r in load_manifest()}
    out = NEW_GOLD_DIR / "corrections.csv"
    # Preserve template rows; fill corrected_text where we have labels
    existing = {}
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                existing[r["line_id"]] = r
    rows = []
    for lid, m in manifest.items():
        prev = existing.get(lid, {})
        corr = latest.get(lid)
        corrected = ""
        model_text = prev.get("model_text") or m.get("model_text") or ""
        if corr:
            if corr["action"] == "marked_illegible":
                corrected = ""
            else:
                corrected = corr.get("corrected_text") or ""
            if corr.get("model_text"):
                model_text = corr["model_text"]
        rows.append(
            {
                "line_id": lid,
                "crop_path": m.get("crop_path") or m.get("image_path") or "",
                "model_text": model_text,
                "corrected_text": corrected,
            }
        )
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["line_id", "crop_path", "model_text", "corrected_text"]
        )
        w.writeheader()
        w.writerows(rows)
    return out


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", default="new_gold")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--blind-count", type=int, default=DEFAULT_BLIND_COUNT)
    p.add_argument("--repeat-rate", type=float, default=DEFAULT_REPEAT_RATE)
    p.add_argument("--prefill-model", default=DEFAULT_PREFILL_MODEL)
    args = p.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing to bind non-local host. Use 127.0.0.1")

    app = create_app(
        queue=args.queue,
        db_path=args.db,
        seed=args.seed,
        blind_count=args.blind_count,
        repeat_rate=args.repeat_rate,
        prefill_model=args.prefill_model,
    )
    # Ensure queue meta + sealed check note
    conn = connect(args.db)
    _ensure_order(
        conn,
        args.queue,
        seed=args.seed,
        blind_count=args.blind_count,
        repeat_rate=args.repeat_rate,
        prefill_model=args.prefill_model,
    )
    seal = NEW_GOLD_DIR / "SEALED.json"
    if seal.exists():
        meta = json.loads(seal.read_text(encoding="utf-8"))
        print(f"[labeling] sealed={meta.get('sealed')} n={meta.get('n_lines')} hash_ok={meta.get('hash_check_ok')}")
    print(f"[labeling] queue={args.queue} db={args.db}")
    print(f"[labeling] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
