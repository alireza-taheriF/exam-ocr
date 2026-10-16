"""Registry activation/rollback and the retrain loop's promotion logic (no model is loaded)."""

from __future__ import annotations

import json
import zipfile

import pytest

from learning import registry, retrain


def _entry(label: str, status: str, adapter: str | None = None, acc: float = 60.0) -> dict:
    return {"label": label, "status": status, "adapter": adapter, "accuracy_pct": acc}


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    monkeypatch.setenv("MODEL_REGISTRY", str(path))
    data = {"champion": _entry("base_v1", "champion"),
            "history": [_entry("base_v1", "champion"), _entry("bad", "rejected")]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_champion_adapter_base_and_missing(reg, tmp_path):
    assert registry.champion_adapter() == ""
    r = registry.load()
    r["champion"]["adapter"] = str(tmp_path / "nope")
    registry.save(r)
    with pytest.raises(FileNotFoundError):
        registry.champion_adapter()
    (tmp_path / "nope").mkdir()
    assert registry.champion_adapter() == str(tmp_path / "nope")


def test_activate_refuses_rejected_and_rollback_restores_previous(reg):
    with pytest.raises(ValueError):
        registry.activate("bad", reason="test")
    r = registry.load()
    r["history"].append(_entry("lora_v2", "champion", acc=80.0))
    r["history"][0]["status"] = "retired"
    r["champion"] = _entry("lora_v2", "champion", acc=80.0)
    registry.save(r)

    assert registry.previous_champion() == "base_v1"
    registry.activate("base_v1", reason="rollback")
    assert registry.load()["champion"]["label"] == "base_v1"
    assert registry.previous_champion() == "lora_v2"


def test_evaluate_rejects_when_holdout_regresses(reg, tmp_path, monkeypatch):
    def fake_gate(cmd, check, cwd):  # eval_gate promoted the candidate on gold
        r = registry.load()
        r["history"].append({**r["champion"], "status": "retired"})
        r["champion"] = _entry("lora_v3", "champion", adapter="models/lora_v3", acc=85.0)
        r["history"].append(dict(r["champion"]))
        registry.save(r)

    monkeypatch.setattr(retrain.subprocess, "run", fake_gate)
    monkeypatch.setattr(retrain, "holdout_check", lambda d, a: {"lines": 40, "candidate_cer": 0.3,
                                                              "served_cer": 0.2, "passed": False})
    retrain.state_path().write_text(json.dumps({"prepared": {"export_dir": str(tmp_path), "corrections_used": 9}}))

    out = retrain.evaluate("lora_v3", "models/lora_v3")
    assert out["status"] == "rejected"
    r = registry.load()
    assert r["champion"]["label"] == "base_v1"
    assert any(e["label"] == "lora_v3" and e["status"] == "rejected" for e in r["history"])


def test_evaluate_promotes_and_records_corrections_used(reg, tmp_path, monkeypatch):
    def fake_gate(cmd, check, cwd):
        r = registry.load()
        r["champion"] = _entry("lora_v3", "champion", adapter="models/lora_v3", acc=85.0)
        r["history"].append(dict(r["champion"]))
        registry.save(r)

    monkeypatch.setattr(retrain.subprocess, "run", fake_gate)
    monkeypatch.setattr(retrain, "holdout_check", lambda d, a: {"lines": 0, "passed": True})
    retrain.state_path().write_text(json.dumps({"prepared": {"export_dir": str(tmp_path), "corrections_used": 9}}))
    out = retrain.evaluate("lora_v3", "models/lora_v3")
    assert out["status"] == "champion"
    assert registry.load()["champion"]["corrections_used"] == 9


def test_prepare_bundles_corrections_with_handwriting(reg, tmp_path, monkeypatch):
    from PIL import Image

    from app.store import Store

    crop = tmp_path / "c.png"
    Image.new("RGB", (120, 30), "white").save(crop)
    hw = tmp_path / "hw.jsonl"
    hw.write_text(json.dumps({"image_path": str(crop), "text": "دست‌خط"}, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(retrain, "HW_MANIFESTS", [hw])
    monkeypatch.setattr(retrain, "EXPORTS", tmp_path / "exports")

    store = Store(tmp_path / "db.sqlite3")
    store.create_document("d1", "a.png", author="s1", course=None, consent=True, holdout=False)
    page = store.add_page("d1", 1, str(crop))
    line = store.add_line("d1", page, 1, kind="text", block_type=None, bbox=[0, 0, 1, 1], crop_path=str(crop))
    store.add_prediction(line, model_version="m", raw_text="اصول هسادجاری", text="اصول هسادجاری", confidence=0.5)
    store.add_edit(line, "اصول حسابداری", author="s1", edit_ms=1)
    store.update_document("d1", confirmed_at=1.0)

    assert retrain.status(store)["trusted_train_corrections"] == 1
    out = retrain.prepare(store, out=tmp_path / "bundle.zip")
    assert out["corrections_used"] == 1
    names = zipfile.ZipFile(tmp_path / "bundle.zip").namelist()
    assert any(n.startswith("manifests/") and "corrections" in n for n in names)
    assert "manifests/hw.jsonl" in names
    assert retrain.load_state()["prepared"]["corrections_used"] == 1
