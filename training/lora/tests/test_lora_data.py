import json

import pytest
import torch

from training.lora import data as D
from training.lora.train_qwen_lora import collate


def _manifest(tmp_path, rows):
    path = tmp_path / "m.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (tmp_path / "a.jpg").write_bytes(b"x")
    return path


def test_load_local_resolves_relative_to_manifest_and_drops_unlabeled(tmp_path):
    m = _manifest(tmp_path, [{"image_path": "a.jpg", "text": "سلام"}, {"image_path": "a.jpg", "text": ""}])
    rows = D.load_local(m, sealed=set())
    assert len(rows) == 1 and rows[0]["image_path"] == str(tmp_path / "a.jpg")


def test_load_local_refuses_sealed_crops(tmp_path):
    m = _manifest(tmp_path, [{"image_path": "a.jpg", "text": "سلام"}])
    with pytest.raises(D.SealedCropError):
        D.load_local(m, sealed={"a.jpg"})


def test_sealed_set_covers_gold_and_test():
    from pathlib import Path

    from training.sealed import sealed_crop_names

    root = Path(__file__).resolve().parents[3]
    gold = [json.loads(l) for l in (root / "dataset" / "gold.jsonl").open(encoding="utf-8") if l.strip()]
    assert {Path(r["image_path"]).name for r in gold} <= sealed_crop_names()


def test_mix_respects_replay_ratio():
    core = [{"image_path": f"c{i}", "text": "t", "source": "local"} for i in range(10)]
    replay = [{"image_path": f"r{i}", "text": "t", "source": "replay"} for i in range(100)]
    rows = D.mix(core[:4], core[4:], replay, replay_ratio=0.5)
    assert sum(r["source"] == "replay" for r in rows) == 5 and len(rows) == 15


def test_manifest_hash_is_order_independent():
    a = [{"image_path": "x/1.jpg", "text": "a"}, {"image_path": "x/2.jpg", "text": "b"}]
    assert D.manifest_hash(a) == D.manifest_hash(list(reversed(a)))


def test_collate_pads_text_and_concatenates_vision_tensors():
    f1 = {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.ones(3, dtype=torch.long),
          "labels": torch.tensor([-100, 2, 3]), "pixel_values": torch.zeros(4, 8),
          "image_grid_thw": torch.tensor([[1, 2, 2]])}
    f2 = {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.ones(2, dtype=torch.long),
          "labels": torch.tensor([-100, 2]), "pixel_values": torch.zeros(6, 8),
          "image_grid_thw": torch.tensor([[1, 2, 3]])}
    b = collate([f1, f2], pad_id=0)
    assert b["input_ids"].tolist() == [[1, 2, 3], [1, 2, 0]]
    assert b["labels"][1, 2].item() == -100 and b["attention_mask"][1, 2].item() == 0
    assert b["pixel_values"].shape == (10, 8) and b["image_grid_thw"].shape == (2, 3)
