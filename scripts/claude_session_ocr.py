#!/usr/bin/env python3
"""Let Claude (in a Claude Code session) read the lines of an uploaded document.

Measured blind on the 240-line gold set, a Claude session read 85.9% against
63.1% for the local Qwen engine, so for documents that matter this is the best
reader available. It is a human-in-the-loop step, not a service: the lines are
packed into numbered sheets, Claude reads each sheet and writes one text file
per sheet, and `apply` stores that text as a new prediction. The student still
edits and confirms in the normal UI.

    1. upload the sheet in the app; wait until the document is "done"
    2. .venv/bin/python scripts/claude_session_ocr.py prepare <doc_id>
    3. in Claude Code: "read the sheets in outputs/claude_ocr/<doc_id>"
       → Claude writes sheet_000.txt, sheet_001.txt, ... (num<TAB>text, ∅ = unreadable)
    4. .venv/bin/python scripts/claude_session_ocr.py apply <doc_id>

Notes:
- The text is stored as a new row in `predictions` (model_version
  "claude-session-opus-5-5"); the earlier engine output is kept.
- Correction-memory rules are NOT applied: they were learned from Qwen's
  mistakes and would corrupt a different reader's output.
- Confidence is stored as NULL (not measured), so the UI flags every line for
  review, which matches ~86% accuracy.
- Unreadable lines (∅) keep the engine's prediction untouched.
- Anything read here is sent to Anthropic through Claude Code. Do this only for
  sheets whose owners agreed (PLAN phase 8: consent for personal data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.store import Store  # noqa: E402
from scripts.claude_label_sheets import make, read_transcriptions  # noqa: E402

MODEL_VERSION = "claude-session-opus-5-5"
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))


def _store(db: Path | None) -> Store:
    return Store(db or DATA_DIR / "exam_ocr.sqlite3")


def _text_lines(store: Store, doc_id: str) -> list[dict]:
    doc = store.get_document(doc_id)
    if doc is None:
        sys.exit(f"document {doc_id} not found")
    if doc["status"] != "done":
        sys.exit(f"document {doc_id} is '{doc['status']}', not 'done'")
    return [l for l in store.document_lines(doc_id) if l["kind"] == "text"]


def prepare(store: Store, doc_id: str, out_dir: Path) -> int:
    lines = _text_lines(store, doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps({"image_path": l["crop_path"], "line_id": l["id"],
                                "source_document": doc_id, "line_index": l["position"]}) + "\n")
    make(manifest, out_dir, allow_sealed=True, limit=0)  # app crops are never part of any sealed set
    return len(lines)


def apply(store: Store, doc_id: str, out_dir: Path, dry_run: bool = False) -> dict:
    doc = store.get_document(doc_id)
    if doc is None:
        sys.exit(f"document {doc_id} not found")
    if doc["confirmed_at"] is not None:
        sys.exit("document is confirmed; its text is locked")
    index = {r["num"]: r for r in map(json.loads, (out_dir / "index.jsonl").open(encoding="utf-8"))}
    texts = read_transcriptions(out_dir)
    valid_ids = {l["id"] for l in _text_lines(store, doc_id)}

    stored = unreadable = 0
    for num, text in sorted(texts.items()):
        line_id = index[num]["line_id"]
        if line_id not in valid_ids:
            sys.exit(f"line {line_id} does not belong to {doc_id}; sheets are from another document?")
        if not text:
            unreadable += 1
            continue
        if not dry_run:
            store.add_prediction(line_id, model_version=MODEL_VERSION, raw_text=text, text=text,
                                 confidence=None, applied_rules=[])
        stored += 1
    return {"lines": len(valid_ids), "read": len(texts), "stored": stored,
            "unreadable_kept_engine_text": unreadable, "not_yet_read": len(valid_ids) - len(texts)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["prepare", "apply"])
    p.add_argument("doc_id")
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="default: outputs/claude_ocr/<doc_id>")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    out = args.out or PROJECT_ROOT / "outputs" / "claude_ocr" / args.doc_id
    store = _store(args.db)
    if args.cmd == "prepare":
        n = prepare(store, args.doc_id, out)
        print(f"{n} text lines ready in {out}")
    else:
        print(json.dumps(apply(store, args.doc_id, out, args.dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
