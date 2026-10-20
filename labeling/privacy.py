"""Privacy gate for every image that could leave this Mac.

    .venv/bin/python -m labeling.privacy            # create/refresh dataset/privacy_allowlist.json, show the 15 largest documents

A document is sendable only if dataset/privacy_allowlist.json marks it {"send": true}. Nothing under eval/ is ever
sendable. `assert_sendable(row)` is the single choke point; `record_sent` keeps the count of images that left.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "dataset" / "privacy_allowlist.json"
SENT_LOG = ROOT / "labeling" / "sent_log.jsonl"


def _rows(name: str) -> list[dict]:
    p = ROOT / "dataset" / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def build() -> dict:
    """Merge the document list into the allowlist; existing `send` flags are kept, new documents default to false."""
    old = json.loads(ALLOWLIST.read_text(encoding="utf-8")) if ALLOWLIST.exists() else {}
    path_split = {r["image_path"]: s for s in ("train", "val", "test") for r in _rows(f"{s}_v3.jsonl")}
    docs: dict[str, dict] = defaultdict(lambda: {"lines": 0, "splits": Counter(), "gold_lines": 0})
    for r in _rows("lines_v3.jsonl"):
        d = docs[r["source_document"]]
        d["lines"] += 1
        d["splits"][path_split.get(r["image_path"], "none")] += 1
    for r in _rows("gold.jsonl"):
        docs[r["source_document"]]["gold_lines"] += 1
    out = {n: {"lines": d["lines"], "splits": dict(d["splits"]), "gold_lines": d["gold_lines"],
               "send": bool(old.get(n, {}).get("send", False))} for n, d in docs.items()}
    ALLOWLIST.write_text(json.dumps(dict(sorted(out.items(), key=lambda kv: -kv[1]["lines"])), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def sendable_docs() -> set[str]:
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8")) if ALLOWLIST.exists() else {}
    return {n for n, d in data.items() if d.get("send") is True}


def assert_sendable(row: dict) -> None:
    """Raise unless this row's image may be sent outside the machine. Call before EVERY hosted-model request."""
    path = str(row.get("image_path") or row.get("source_image") or "")
    if "eval" in Path(path).parts:
        raise PermissionError(f"{path}: nothing under eval/ may be sent")
    if row.get("source_document") not in sendable_docs():
        raise PermissionError(f"{path}: document {row.get('source_document')!r} is not send:true in {ALLOWLIST.name}")


def record_sent(row: dict, model: str) -> None:
    with SENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"image_path": row.get("image_path"), "model": model, "ts": int(time.time())}) + "\n")


def sent_count() -> int:
    return len(SENT_LOG.read_text().splitlines()) if SENT_LOG.exists() else 0


if __name__ == "__main__":
    data = build()
    print(f"{len(data)} documents -> {ALLOWLIST.relative_to(ROOT)}   (send:true = {len(sendable_docs())})")
    print("| # | lines | train/val/test | gold | send | document |\n|---|---|---|---|---|---|")
    for i, (n, d) in enumerate(list(json.loads(ALLOWLIST.read_text(encoding="utf-8")).items())[:15], 1):
        s = d["splits"]
        print(f"| {i} | {d['lines']} | {s.get('train', 0)}/{s.get('val', 0)}/{s.get('test', 0)} | {d['gold_lines']} | {d['send']} | {n[:60]} |")
    print("gold documents:", [n for n, d in data.items() if d["gold_lines"]])
