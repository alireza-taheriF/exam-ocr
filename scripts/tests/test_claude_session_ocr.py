import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.store import Store  # noqa: E402
from scripts import claude_session_ocr as cso  # noqa: E402


@pytest.fixture
def doc(tmp_path):
    store = Store(tmp_path / "t.sqlite3")
    store.create_document("d1", "a.png", author=None, course=None, consent=True, holdout=False)
    page = store.add_page("d1", 1, str(tmp_path / "page.png"))
    ids = []
    for pos, kind in enumerate(["text", "text", "image", "text"], start=1):
        crop = tmp_path / f"{pos}.png"
        Image.new("RGB", (300, 40), "white").save(crop)
        lid = store.add_line("d1", page, pos, kind=kind, block_type=None, bbox=[0, 0, 300, 40], crop_path=str(crop))
        if kind == "text":
            store.add_prediction(lid, model_version="qwen", raw_text="qwen", text="qwen", confidence=0.9)
        ids.append(lid)
    store.update_document("d1", status="done")
    return store, ids, tmp_path


def test_prepare_lists_only_text_lines_with_line_ids(doc):
    store, ids, tmp = doc
    out = tmp / "sheets"
    assert cso.prepare(store, "d1", out) == 3
    import json
    index = [json.loads(l) for l in (out / "index.jsonl").open()]
    assert [r["line_id"] for r in index] == [ids[0], ids[1], ids[3]]  # the image block is skipped


def test_apply_stores_new_prediction_and_keeps_unreadable(doc):
    store, ids, tmp = doc
    out = tmp / "sheets"
    cso.prepare(store, "d1", out)
    (out / "sheet_000.txt").write_text("0\tسلام دنیا\n1\t∅\n2\tکتاب\n", encoding="utf-8")
    r = cso.apply(store, "d1", out)
    assert r == {"lines": 3, "read": 3, "stored": 2, "unreadable_kept_engine_text": 1, "not_yet_read": 0}
    p0 = store.latest_prediction(ids[0])
    assert (p0["text"], p0["model_version"], p0["confidence"], p0["applied_rules"]) == (
        "سلام دنیا", cso.MODEL_VERSION, None, "[]")
    assert store.latest_prediction(ids[1])["text"] == "qwen"       # ∅ keeps the engine text
    assert store.latest_prediction(ids[3])["text"] == "کتاب"


def test_apply_refuses_confirmed_document_and_dry_run_writes_nothing(doc):
    store, ids, tmp = doc
    out = tmp / "sheets"
    cso.prepare(store, "d1", out)
    (out / "sheet_000.txt").write_text("0\tx\n", encoding="utf-8")
    cso.apply(store, "d1", out, dry_run=True)
    assert store.latest_prediction(ids[0])["text"] == "qwen"
    store.update_document("d1", confirmed_at=1.0)
    with pytest.raises(SystemExit):
        cso.apply(store, "d1", out)
