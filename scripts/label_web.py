#!/usr/bin/env python3
"""Local labeling page: crop on top, draft transcription prefilled, fix it, Enter, next.

    python scripts/label_web.py --drafts htr_eval/cache/qwen3vl-fa.jsonl      # then open http://127.0.0.1:8765

Keys: Enter = save + next | Ctrl+Backspace = unreadable (empty label, skipped in training)
      Ctrl+Z (text untouched) = undo last decision.
Labels are LITERAL (author's spelling and digits), same convention as label_gold_interactive.py.
Every decision is appended to dataset/human_labels.jsonl at once; restarting resumes at the first
unlabeled queue line. Binds 127.0.0.1 only; serves only images listed in the queue.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE, LABELS = ROOT / "dataset" / "label_queue.jsonl", ROOT / "dataset" / "human_labels.jsonl"

PAGE = """<!doctype html><meta charset=utf-8><title>label</title>
<style>body{font:16px system-ui;max-width:1100px;margin:16px auto;padding:0 16px}
img{width:100%;border:1px solid #888;background:#fff}
#t{width:100%;font-size:30px;direction:rtl;padding:8px;box-sizing:border-box}
#bar{display:flex;justify-content:space-between;color:#555;margin:8px 0}</style>
<div id=bar><span id=prog></span><span id=med></span></div>
<img id=im><input id=t autocomplete=off autofocus>
<p style="color:#666">Enter = save · Ctrl+Backspace = unreadable · Ctrl+Z = undo (text untouched) · literal transcription: keep the author's spelling and digits</p>
<script>
let cur=null,t0=0,draft="";const $=i=>document.getElementById(i);
async function load(){const r=await (await fetch('/api/next')).json();cur=r;
 if(r.finished){document.body.innerHTML='<h2>All '+r.total+' done. Tell Claude "done".</h2>';return}
 $('prog').textContent=r.done+' / '+r.total+'   '+r.source_document;
 $('med').textContent='median '+(r.median??'-')+' s/line';
 $('im').src='/img/'+r.index;draft=r.draft;$('t').value=draft;$('t').focus();t0=performance.now()}
async function send(unreadable){const s=(performance.now()-t0)/1000;
 await fetch('/api/save',{method:'POST',body:JSON.stringify({index:cur.index,text:unreadable?'':$('t').value.trim(),unreadable,seconds:s})});load()}
$('t').addEventListener('keydown',async e=>{
 if(e.key==='Enter'){e.preventDefault();send(false)}
 else if(e.key==='Backspace'&&e.ctrlKey){e.preventDefault();send(true)}
 else if(e.key==='z'&&e.ctrlKey&&$('t').value===draft){e.preventDefault();await fetch('/api/undo',{method:'POST'});load()}});
load();
</script>"""


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def state() -> dict[str, dict]:
    """Replay the append-only log: later decisions win, an undo row drops the line."""
    out: dict[str, dict] = {}
    for r in read_jsonl(LABELS):
        if r.get("undo"):
            out.pop(r["image_path"], None)
        else:
            out[r["image_path"]] = r
    return out


def make_handler(queue: list[dict], drafts):  # drafts: callable returning {image_path: text}
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            if self.path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/img/"):
                try:
                    p = ROOT / queue[int(self.path[5:])]["image_path"]
                    return self._send(200, p.read_bytes(), "image/jpeg")
                except (ValueError, IndexError, OSError):
                    return self._send(404, b"", "text/plain")
            if self.path == "/api/next":
                st = state()
                secs = [r["seconds"] for r in st.values() if r.get("seconds")]
                med = round(statistics.median(secs), 1) if secs else None
                for i, q in enumerate(queue):
                    if q["image_path"] not in st:
                        return self._json({"index": i, "done": len(st), "total": len(queue), "median": med,
                                           "source_document": q["source_document"][:40],
                                           "draft": drafts().get(q["image_path"], "")})
                return self._json({"finished": True, "total": len(queue)})
            self._send(404, b"", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/save":
                q = queue[int(body["index"])]
                row = {"image_path": q["image_path"], "source_document": q["source_document"],
                       "text": str(body["text"]), "unreadable": bool(body["unreadable"]),
                       "seconds": round(float(body["seconds"]), 2), "ts": int(time.time())}
            elif self.path == "/api/undo":
                done = [r for r in read_jsonl(LABELS) if not r.get("undo")]
                st = state()
                last = next((r for r in reversed(done) if r["image_path"] in st), None)
                if not last:
                    return self._json({"ok": False})
                row = {"image_path": last["image_path"], "undo": True, "ts": int(time.time())}
            else:
                return self._send(404, b"", "text/plain")
            with LABELS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._json({"ok": True})

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drafts", type=Path, required=True, help="predictions JSONL used to prefill the text field")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    queue = read_jsonl(QUEUE)
    path = ROOT / args.drafts

    def drafts() -> dict[str, str]:  # re-read each time: the draft run may still be producing lines
        return {r["image_path"]: r.get("text", "") for r in read_jsonl(path)}

    print(f"{len(queue)} lines, {len(state())} already labeled, {len(drafts())} drafts -> http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(queue, drafts)).serve_forever()


if __name__ == "__main__":
    main()
