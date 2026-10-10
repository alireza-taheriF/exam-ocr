"""SQLite persistence for documents, line crops, OCR predictions and student edits.

Edits are append-only: every save adds a new row in `edits` with an increasing
`version`, so the original OCR text and the full edit history of each line are
always recoverable (required for exam answer sheets, see PLAN §3 phase 8).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    author        TEXT,
    course        TEXT,
    consent       INTEGER NOT NULL DEFAULT 0,
    holdout       INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'queued',
    step          TEXT,
    error         TEXT,
    model_version TEXT,
    created_at    REAL NOT NULL,
    confirmed_at  REAL
);

CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index  INTEGER NOT NULL,
    image_path  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id     INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,
    kind        TEXT NOT NULL,          -- 'text' (editable) | 'image' (table/figure, crop only)
    block_type  TEXT,
    bbox        TEXT NOT NULL,          -- JSON [x1, y1, x2, y2] in page pixels
    crop_path   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id       INTEGER NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    raw_text      TEXT NOT NULL,        -- engine output before correction memory
    text          TEXT NOT NULL,        -- what the student was shown
    confidence    REAL,
    applied_rules TEXT NOT NULL DEFAULT '[]',
    error         TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS edits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id       INTEGER NOT NULL REFERENCES lines(id) ON DELETE CASCADE,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    version       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    author        TEXT,
    edit_ms       INTEGER,
    created_at    REAL NOT NULL,
    UNIQUE (line_id, version)
);

CREATE INDEX IF NOT EXISTS idx_lines_doc ON lines(document_id, position);
CREATE INDEX IF NOT EXISTS idx_pred_line ON predictions(line_id);
CREATE INDEX IF NOT EXISTS idx_edits_line ON edits(line_id, version);
"""


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # ── documents ───────────────────────────────────────────────────────────
    def create_document(
        self,
        doc_id: str,
        filename: str,
        *,
        author: str | None,
        course: str | None,
        consent: bool,
        holdout: bool,
    ) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO documents (id, filename, author, course, consent, holdout, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, filename, author, course, int(consent), int(holdout), time.time()),
            )

    def update_document(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._tx() as c:
            c.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM documents WHERE id = ?", (doc_id,))

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._all(
            "SELECT d.*,"
            " (SELECT COUNT(*) FROM lines l WHERE l.document_id = d.id AND l.kind = 'text') AS text_lines,"
            " (SELECT COUNT(DISTINCT e.line_id) FROM edits e JOIN lines l ON l.id = e.line_id"
            "   WHERE l.document_id = d.id) AS edited_lines"
            " FROM documents d ORDER BY d.created_at DESC LIMIT ?",
            (limit,),
        )

    # ── pages / lines / predictions ─────────────────────────────────────────
    def add_page(self, doc_id: str, page_index: int, image_path: str) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO pages (document_id, page_index, image_path) VALUES (?, ?, ?)",
                (doc_id, page_index, image_path),
            )
            return int(cur.lastrowid)

    def add_line(
        self,
        doc_id: str,
        page_id: int,
        position: int,
        *,
        kind: str,
        block_type: str | None,
        bbox: list[int],
        crop_path: str,
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO lines (document_id, page_id, position, kind, block_type, bbox, crop_path)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (doc_id, page_id, position, kind, block_type, json.dumps(bbox), crop_path),
            )
            return int(cur.lastrowid)

    def add_prediction(
        self,
        line_id: int,
        *,
        model_version: str,
        raw_text: str,
        text: str,
        confidence: float | None,
        applied_rules: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO predictions"
                " (line_id, model_version, raw_text, text, confidence, applied_rules, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    line_id,
                    model_version,
                    raw_text,
                    text,
                    confidence,
                    json.dumps(applied_rules or [], ensure_ascii=False),
                    error,
                    time.time(),
                ),
            )
            return int(cur.lastrowid)

    def get_line(self, line_id: int) -> dict[str, Any] | None:
        return self._one("SELECT * FROM lines WHERE id = ?", (line_id,))

    def latest_prediction(self, line_id: int) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM predictions WHERE line_id = ? ORDER BY id DESC LIMIT 1", (line_id,)
        )

    def document_lines(self, doc_id: str) -> list[dict[str, Any]]:
        """Lines in reading order with latest prediction and latest edit joined in."""
        rows = self._all(
            "SELECT l.*, p.id AS prediction_id, p.model_version, p.raw_text, p.text AS predicted_text,"
            "       p.confidence, p.applied_rules, p.error AS prediction_error,"
            "       e.text AS edited_text, e.version AS edit_version, e.author AS edit_author"
            " FROM lines l"
            " LEFT JOIN predictions p ON p.id = (SELECT MAX(id) FROM predictions WHERE line_id = l.id)"
            " LEFT JOIN edits e ON e.id = (SELECT MAX(id) FROM edits WHERE line_id = l.id)"
            " WHERE l.document_id = ? ORDER BY l.position",
            (doc_id,),
        )
        for row in rows:
            row["bbox"] = json.loads(row["bbox"])
            row["applied_rules"] = json.loads(row["applied_rules"] or "[]")
        return rows

    # ── edits (append-only) ─────────────────────────────────────────────────
    def add_edit(
        self, line_id: int, text: str, *, author: str | None, edit_ms: int | None
    ) -> dict[str, Any]:
        with self._tx() as c:
            pred = c.execute(
                "SELECT id, text FROM predictions WHERE line_id = ? ORDER BY id DESC LIMIT 1",
                (line_id,),
            ).fetchone()
            if pred is None:
                raise ValueError(f"line {line_id} has no prediction to edit")
            last = c.execute(
                "SELECT version, text FROM edits WHERE line_id = ? ORDER BY version DESC LIMIT 1",
                (line_id,),
            ).fetchone()
            current = last["text"] if last else pred["text"]
            if last is not None and current == text:
                return {"version": last["version"], "changed": False}
            version = (last["version"] if last else 0) + 1
            c.execute(
                "INSERT INTO edits (line_id, prediction_id, version, text, author, edit_ms, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (line_id, pred["id"], version, text, author, edit_ms, time.time()),
            )
            return {"version": version, "changed": True}

    def line_history(self, line_id: int) -> dict[str, Any]:
        return {
            "predictions": self._all(
                "SELECT id, model_version, raw_text, text, confidence, applied_rules, created_at"
                " FROM predictions WHERE line_id = ? ORDER BY id",
                (line_id,),
            ),
            "edits": self._all(
                "SELECT version, text, author, edit_ms, created_at FROM edits"
                " WHERE line_id = ? ORDER BY version",
                (line_id,),
            ),
        }

    def final_pairs(self, *, confirmed_only: bool = False) -> list[dict[str, Any]]:
        """One row per edited text line: shown prediction vs. latest student text."""
        sql = (
            "SELECT l.id AS line_id, l.document_id, l.crop_path, d.author AS doc_author,"
            "       d.course, d.consent, d.holdout, d.confirmed_at,"
            "       p.model_version, p.raw_text, p.text AS predicted_text, p.confidence,"
            "       e.text AS final_text, e.author, e.version"
            " FROM edits e"
            " JOIN lines l ON l.id = e.line_id"
            " JOIN documents d ON d.id = l.document_id"
            " JOIN predictions p ON p.id = e.prediction_id"
            " WHERE e.id = (SELECT MAX(id) FROM edits WHERE line_id = e.line_id)"
        )
        if confirmed_only:
            sql += " AND d.confirmed_at IS NOT NULL"
        return self._all(sql + " ORDER BY e.line_id")

    def confirmed_untouched_pairs(self) -> list[dict[str, Any]]:
        """Text lines of confirmed documents that the student accepted without editing."""
        return self._all(
            "SELECT l.id AS line_id, l.document_id, l.crop_path, d.author AS doc_author, d.course,"
            "       d.consent, d.holdout, d.confirmed_at, p.model_version, p.raw_text,"
            "       p.text AS predicted_text, p.confidence, p.text AS final_text,"
            "       d.author AS author, 0 AS version"
            " FROM lines l JOIN documents d ON d.id = l.document_id"
            " JOIN predictions p ON p.id = (SELECT MAX(id) FROM predictions WHERE line_id = l.id)"
            " WHERE l.kind = 'text' AND d.confirmed_at IS NOT NULL"
            "   AND NOT EXISTS (SELECT 1 FROM edits e WHERE e.line_id = l.id)"
            "   AND p.text != ''"
        )
