import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from scripts import label_web


def call(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=None if body is None else json.dumps(body).encode())
    return json.loads(urllib.request.urlopen(req).read())


def test_save_resume_undo(tmp_path, monkeypatch):
    monkeypatch.setattr(label_web, "LABELS", tmp_path / "labels.jsonl")
    queue = [{"image_path": f"a{i}.jpg", "source_document": "d"} for i in range(3)]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), label_web.make_handler(queue, lambda: {"a0.jpg": "پیش‌نویس"}))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    assert call(port, "/api/next")["draft"] == "پیش‌نویس"
    call(port, "/api/save", {"index": 0, "text": "سلام", "unreadable": False, "seconds": 4.0})
    call(port, "/api/save", {"index": 1, "text": "", "unreadable": True, "seconds": 2.0})
    assert call(port, "/api/next")["index"] == 2 and call(port, "/api/next")["median"] == 3.0
    call(port, "/api/undo", {})
    assert call(port, "/api/next")["index"] == 1  # the unreadable decision was undone
    srv.shutdown()
