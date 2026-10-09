"""SQLite storage for corrections (append-only) and cached model preds."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "labeling" / "labels.db"
DEFAULT_JSONL = PROJECT_ROOT / "labeling" / "exports" / "corrections.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id TEXT NOT NULL,
    queue TEXT NOT NULL,
    model_text TEXT,
    corrected_text TEXT,
    action TEXT NOT NULL,
    region_type TEXT NOT NULL DEFAULT 'body',
    reviewer TEXT,
    started_at TEXT,
    saved_at TEXT NOT NULL,
    seconds_spent REAL,
    notes TEXT,
    blind_mode INTEGER NOT NULL DEFAULT 0,
    presentation_index INTEGER,
    is_repeat INTEGER NOT NULL DEFAULT 0,
    prefill_model TEXT
);

CREATE INDEX IF NOT EXISTS idx_corrections_line_queue
    ON corrections(queue, line_id, id);

CREATE TABLE IF NOT EXISTS model_preds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    line_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(line_id, model_id)
);

CREATE TABLE IF NOT EXISTS queue_meta (
    queue TEXT PRIMARY KEY,
    seed INTEGER NOT NULL,
    blind_count INTEGER NOT NULL,
    repeat_rate REAL NOT NULL,
    order_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

ACTIONS = frozenset(
    {"accepted_as_is", "edited", "marked_illegible", "skipped"}
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_model_pred(
    conn: sqlite3.Connection,
    *,
    line_id: str,
    model_id: str,
    text: str,
) -> None:
    conn.execute(
        """
        INSERT INTO model_preds(line_id, model_id, text, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(line_id, model_id) DO UPDATE SET
            text=excluded.text,
            created_at=excluded.created_at
        """,
        (line_id, model_id, text, utcnow()),
    )
    conn.commit()


def get_model_preds(conn: sqlite3.Connection, line_id: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT model_id, text FROM model_preds WHERE line_id=?",
        (line_id,),
    ).fetchall()
    return {r["model_id"]: r["text"] for r in rows}


def count_model_preds(conn: sqlite3.Connection, model_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM model_preds WHERE model_id=?",
        (model_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def save_queue_meta(
    conn: sqlite3.Connection,
    *,
    queue: str,
    seed: int,
    blind_count: int,
    repeat_rate: float,
    order: list[dict],
) -> None:
    conn.execute(
        """
        INSERT INTO queue_meta(queue, seed, blind_count, repeat_rate, order_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(queue) DO UPDATE SET
            seed=excluded.seed,
            blind_count=excluded.blind_count,
            repeat_rate=excluded.repeat_rate,
            order_json=excluded.order_json,
            updated_at=excluded.updated_at
        """,
        (queue, seed, blind_count, repeat_rate, json.dumps(order, ensure_ascii=False), utcnow()),
    )
    conn.commit()


def load_queue_meta(conn: sqlite3.Connection, queue: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM queue_meta WHERE queue=?", (queue,)
    ).fetchone()
    if not row:
        return None
    return {
        "queue": row["queue"],
        "seed": row["seed"],
        "blind_count": row["blind_count"],
        "repeat_rate": row["repeat_rate"],
        "order": json.loads(row["order_json"]),
        "updated_at": row["updated_at"],
    }


def append_correction(
    conn: sqlite3.Connection,
    *,
    line_id: str,
    queue: str,
    model_text: str,
    corrected_text: str,
    action: str,
    region_type: str,
    reviewer: str,
    started_at: str | None,
    seconds_spent: float | None,
    notes: str,
    blind_mode: bool,
    presentation_index: int | None,
    is_repeat: bool,
    prefill_model: str,
    jsonl_path: Path | None = None,
) -> int:
    if action not in ACTIONS:
        raise ValueError(f"invalid action: {action}")
    saved_at = utcnow()
    cur = conn.execute(
        """
        INSERT INTO corrections(
            line_id, queue, model_text, corrected_text, action, region_type,
            reviewer, started_at, saved_at, seconds_spent, notes,
            blind_mode, presentation_index, is_repeat, prefill_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            line_id,
            queue,
            model_text,
            corrected_text,
            action,
            region_type,
            reviewer,
            started_at,
            saved_at,
            seconds_spent,
            notes,
            1 if blind_mode else 0,
            presentation_index,
            1 if is_repeat else 0,
            prefill_model,
        ),
    )
    conn.commit()
    row_id = int(cur.lastrowid)
    export_row = {
        "id": row_id,
        "line_id": line_id,
        "queue": queue,
        "model_text": model_text,
        "corrected_text": corrected_text,
        "action": action,
        "region_type": region_type,
        "reviewer": reviewer,
        "started_at": started_at,
        "saved_at": saved_at,
        "seconds_spent": seconds_spent,
        "notes": notes,
        "blind_mode": blind_mode,
        "presentation_index": presentation_index,
        "is_repeat": is_repeat,
        "prefill_model": prefill_model,
    }
    path = Path(jsonl_path or DEFAULT_JSONL)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(export_row, ensure_ascii=False) + "\n")
    return row_id


def latest_corrections(
    conn: sqlite3.Connection, queue: str
) -> dict[str, dict[str, Any]]:
    """Map line_id -> latest correction row (by id)."""
    rows = conn.execute(
        """
        SELECT c.* FROM corrections c
        INNER JOIN (
            SELECT line_id, MAX(id) AS max_id
            FROM corrections WHERE queue=?
            GROUP BY line_id
        ) t ON c.id = t.max_id
        """,
        (queue,),
    ).fetchall()
    return {r["line_id"]: dict(r) for r in rows}


def latest_non_skip(
    conn: sqlite3.Connection, queue: str
) -> dict[str, dict[str, Any]]:
    """Latest meaningful correction per line_id, honoring append-only __UNDO__."""
    rows = conn.execute(
        "SELECT * FROM corrections WHERE queue=? ORDER BY id ASC",
        (queue,),
    ).fetchall()
    stacks: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        lid = r["line_id"]
        stacks.setdefault(lid, [])
        if (r["notes"] or "") == "__UNDO__":
            if stacks[lid]:
                stacks[lid].pop()
            continue
        if r["action"] == "skipped" and (r["notes"] or "") != "__UNDO__":
            # plain skip still records a presentation; keep for line-level label? no.
            continue
        stacks[lid].append(dict(r))
    return {lid: stack[-1] for lid, stack in stacks.items() if stack}


def last_correction_row(conn: sqlite3.Connection, queue: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM corrections WHERE queue=? ORDER BY id DESC LIMIT 1",
        (queue,),
    ).fetchone()
    return dict(row) if row else None


def undo_last(conn: sqlite3.Connection, queue: str) -> dict | None:
    """Soft-undo: append a notes marker and delete is wrong for append-only.

    Append-only policy: we insert an action='skipped' tombstone is not ideal.
    Instead delete is forbidden — we mark by inserting a row with
    action='skipped' and notes='UNDO_PREV' only for UI cursor; the previous
    content remains in history. For resume we treat 'undone' via a dedicated
    approach: store undo as appending notes on a new row that copies previous
    with action skipped and notes=__UNDO__.

    Simpler approach used here: physically delete only the latest row id for
    this queue AND rewrite JSONL from DB (history still mostly append via
    exports rebuild). User asked append-only history — so we append an
    undo record instead.
    """
    last = last_correction_row(conn, queue)
    if not last:
        return None
    # Append undo sentinel so history stays append-only.
    undo_id = append_correction(
        conn,
        line_id=last["line_id"],
        queue=queue,
        model_text=last.get("model_text") or "",
        corrected_text="",
        action="skipped",
        region_type=last.get("region_type") or "body",
        reviewer=last.get("reviewer") or "",
        started_at=None,
        seconds_spent=0.0,
        notes="__UNDO__",
        blind_mode=bool(last.get("blind_mode")),
        presentation_index=last.get("presentation_index"),
        is_repeat=bool(last.get("is_repeat")),
        prefill_model=last.get("prefill_model") or "",
    )
    return {"undone_id": last["id"], "undo_row_id": undo_id, "line_id": last["line_id"]}


def iter_history(conn: sqlite3.Connection, queue: str) -> Iterable[dict]:
    rows = conn.execute(
        "SELECT * FROM corrections WHERE queue=? ORDER BY id ASC",
        (queue,),
    ).fetchall()
    for r in rows:
        yield dict(r)


def rebuild_jsonl(
    conn: sqlite3.Connection,
    queue: str,
    path: Path | None = None,
) -> Path:
    out = Path(path or DEFAULT_JSONL)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in iter_history(conn, queue):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
