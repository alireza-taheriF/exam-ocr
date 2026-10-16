"""End-to-end API flow with the fake OCR engine and the OpenCV segmenter."""

from __future__ import annotations

import importlib
import io
import sys

import cv2
import numpy as np
import pytest


def _page_png(lines: int = 5) -> bytes:
    img = np.full((1100, 850, 3), 255, dtype=np.uint8)
    for i in range(lines):
        y = 120 + i * 110
        cv2.putText(img, "Answer line number %d text" % i, (90, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    cv2.rectangle(img, (300, 760), (560, 1020), (0, 0, 0), 3)   # a drawing → image item
    cv2.circle(img, (430, 890), 90, (0, 0, 0), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OCR_ENGINE", "fake")
    monkeypatch.setenv("SEGMENTER", "opencv")
    monkeypatch.setenv("HOLDOUT_FRACTION", "0")
    monkeypatch.setenv("MEMORY_MIN_AUTHORS", "2")
    from pipeline.engines import get_engine

    get_engine.cache_clear()
    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c
    sys.modules.pop("main", None)


def _upload(client, author: str, consent: bool = True) -> dict:
    r = client.post(
        "/api/documents",
        files={"file": ("sheet.png", io.BytesIO(_page_png()), "image/png")},
        data={"author": author, "course": "حسابداری ۱", "consent": str(consent).lower()},
    )
    assert r.status_code == 202, r.text
    doc = client.get(f"/api/documents/{r.json()['id']}").json()
    assert doc["status"] == "done", doc
    return doc


def _fix_all(client, doc: dict, author: str) -> None:
    for line in doc["lines"]:
        if line["kind"] == "text":
            fixed = line["text"].replace("هسادجاری", "حسابداری")
            r = client.put(f"/api/lines/{line['id']}", json={"text": fixed, "author": author, "edit_ms": 900})
            assert r.status_code == 200, r.text
    assert client.post(f"/api/documents/{doc['id']}/confirm").status_code == 200


def test_upload_produces_editable_lines_and_image_items(client):
    doc = _upload(client, "s1")
    kinds = [l["kind"] for l in doc["lines"]]
    assert kinds.count("text") >= 4
    assert "image" in kinds
    first = next(l for l in doc["lines"] if l["kind"] == "text")
    assert first["text"] == "اصول هسادجاری و معادله اساسی"
    assert client.get(first["crop_url"]).headers["content-type"] == "image/png"


def test_edit_history_is_append_only_and_locked_after_confirm(client):
    doc = _upload(client, "s1")
    line = next(l for l in doc["lines"] if l["kind"] == "text")
    url = f"/api/lines/{line['id']}"
    assert client.put(url, json={"text": "اول"}).json()["version"] == 1
    assert client.put(url, json={"text": "اول"}).json()["changed"] is False
    assert client.put(url, json={"text": "دوم"}).json()["version"] == 2

    hist = client.get(f"{url}/history").json()
    assert [e["text"] for e in hist["edits"]] == ["اول", "دوم"]
    assert hist["predictions"][0]["raw_text"] == "اصول هسادجاری و معادله اساسی"

    client.post(f"/api/documents/{doc['id']}/confirm")
    assert client.put(url, json={"text": "سوم"}).status_code == 409


def test_correction_memory_learns_from_independent_students(client):
    _fix_all(client, _upload(client, "s1"), "s1")
    # One student is not enough (MEMORY_MIN_AUTHORS=2).
    doc2 = _upload(client, "s2")
    assert "هسادجاری" in doc2["lines"][0]["text"] or doc2["lines"][0]["kind"] == "image"
    _fix_all(client, doc2, "s2")

    doc3 = _upload(client, "s3")
    texts = [l["text"] for l in doc3["lines"] if l["kind"] == "text"]
    assert all("حسابداری" in t and "هسادجاری" not in t for t in texts)
    line = next(l for l in doc3["lines"] if l["kind"] == "text")
    assert line["raw_text"] == "اصول هسادجاری و معادله اساسی"
    assert line["applied_rules"][0]["dst"] == "حسابداری"

    stats = client.get("/api/stats").json()
    assert any(r["src"] == "هسادجاری" for r in stats["rules"])
    assert stats["versions"][0]["post_edit_accuracy_pct"] < 100


def test_no_consent_means_no_learning(client):
    for s in ("s1", "s2"):
        _fix_all(client, _upload(client, s, consent=False), s)
    doc = _upload(client, "s3")
    assert all("هسادجاری" in l["text"] for l in doc["lines"] if l["kind"] == "text")


def test_export_uses_edited_text(client):
    doc = _upload(client, "s1")
    line = next(l for l in doc["lines"] if l["kind"] == "text")
    client.put(f"/api/lines/{line['id']}", json={"text": "متن اصلاح‌شده"})
    txt = client.get(f"/api/documents/{doc['id']}/export?format=txt").text
    assert "متن اصلاح‌شده" in txt
    r = client.get(f"/api/documents/{doc['id']}/export?format=docx")
    assert r.status_code == 200 and r.content[:2] == b"PK"


def test_rejects_bad_extension(client):
    r = client.post("/api/documents", files={"file": ("a.exe", io.BytesIO(b"x"), "application/octet-stream")})
    assert r.status_code == 400


def test_edit_button_keeps_image_and_shows_it_to_admin(client):
    doc = _upload(client, "s1")
    line = next(l for l in doc["lines"] if l["kind"] == "text")
    r = client.post(f"/api/lines/{line['id']}/edit", json={"author": "s1"})
    assert r.status_code == 200, r.text
    assert r.json()["shown_text"] == line["text"]
    client.put(f"/api/lines/{line['id']}", json={"text": "اصول حسابداری", "author": "s1"})

    rows = client.get("/api/admin/corrections").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["predicted_text"] == "اصول هسادجاری و معادله اساسی"
    assert row["student_text"] == "اصول حسابداری"
    assert row["author"] == "s1"
    img = client.get(row["image_url"])
    assert img.status_code == 200
    assert img.content == client.get(line["crop_url"]).content


def test_save_without_edit_button_still_keeps_image(client):
    doc = _upload(client, "s1")
    line = next(l for l in doc["lines"] if l["kind"] == "text")
    client.put(f"/api/lines/{line['id']}", json={"text": "متن"})
    rows = client.get("/api/admin/corrections").json()
    assert [r["line_id"] for r in rows] == [line["id"]]
    assert client.get(rows[0]["image_url"]).status_code == 200


def test_edit_button_rejected_after_confirm_and_for_image_items(client):
    doc = _upload(client, "s1")
    image = next(l for l in doc["lines"] if l["kind"] == "image")
    assert client.post(f"/api/lines/{image['id']}/edit", json={}).status_code == 400
    client.post(f"/api/documents/{doc['id']}/confirm")
    text = next(l for l in doc["lines"] if l["kind"] == "text")
    assert client.post(f"/api/lines/{text['id']}/edit", json={}).status_code == 409


def test_admin_review_overrides_trust_for_learning(client):
    # Two students rewrite every word: near-total rewrite is quarantined, so no rule is learned.
    docs = []
    for s in ("s1", "s2"):
        doc = _upload(client, s)
        line = next(l for l in doc["lines"] if l["kind"] == "text" and not l["low_confidence"])
        client.put(f"/api/lines/{line['id']}", json={"text": "ثثثث ثثثث ث ثثثثث ثثثثث", "author": s})
        client.post(f"/api/documents/{doc['id']}/confirm")
        docs.append(doc)
    rows = client.get("/api/admin/corrections").json()
    assert {r["trust"] for r in rows} == {"quarantined"}
    assert client.get("/api/stats").json()["candidates"] == []

    for r in rows:
        assert client.post(f"/api/admin/lines/{r['line_id']}/review", json={"verdict": "approved"}).status_code == 200
    rows = client.get("/api/admin/corrections").json()
    assert {r["trust"] for r in rows} == {"trusted"}
    assert {r["review"] for r in rows} == {"approved"}
    # Approved corrections now reach the memory (as candidates: the unedited lines of the
    # same documents still count as evidence against the substitution).
    assert any(c["dst"] == "ثثثث" and c["authors"] == 2 for c in client.get("/api/stats").json()["candidates"])

    client.post(f"/api/admin/lines/{rows[0]['line_id']}/review", json={"verdict": "rejected"})
    assert client.get("/api/admin/corrections").json()[0]["trust"] == "rejected"
    assert client.post(f"/api/admin/lines/{rows[0]['line_id']}/review", json={"verdict": "maybe"}).status_code == 422


def test_retrain_status_counts_trusted_corrections(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_REGISTRY", str(tmp_path / "registry.json"))
    monkeypatch.setenv("RETRAIN_MIN_CORRECTIONS", "3")
    _fix_all(client, _upload(client, "s1"), "s1")
    r = client.get("/api/admin/retrain").json()
    assert r["trusted_train_corrections"] >= 4
    assert r["ready"] is True
    assert client.post("/api/admin/reload-engine").json()["engine"] == "fake@1"
