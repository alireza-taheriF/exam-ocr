"""Tests for scripts/split_dataset.py — the fix for Blocker B (split leakage).

The property under test is the one the legacy script violated: no source document
may appear in more than one split. These tests assert it directly rather than
trusting the report the script prints about itself.
"""

import pytest

from scripts.tests.conftest import load_script

sd = load_script("split_under_test", "scripts/split_dataset.py")


def rows_for(doc: str, n: int, *, pages: int = 1, text: str = "متن") -> list[dict]:
    return [
        {
            "image_path": f"dataset/lines_v2/{doc}_{i}.jpg",
            "text": f"{text} {i}",
            "source_image": f"dataset/raw_images/{doc}_p{i % pages:03d}.png",
            "source_document": doc,
            "line_index": i,
            "bbox": [0, 0, 100, 40],
        }
        for i in range(n)
    ]


# ── The invariant: documents never cross splits ──────────────────────────────


def test_no_document_appears_in_two_splits():
    groups = {f"doc{i}": rows_for(f"doc{i}", 10 * (i + 1)) for i in range(12)}
    assignment = sd.assign_groups(
        groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
    )
    assert set(assignment) == set(groups)
    per_split = {s: set() for s in sd.SPLITS}
    for doc, split in assignment.items():
        per_split[split].add(doc)
    assert per_split["train"] & per_split["val"] == set()
    assert per_split["train"] & per_split["test"] == set()
    assert per_split["val"] & per_split["test"] == set()


def test_every_group_is_assigned_exactly_once():
    groups = {f"doc{i}": rows_for(f"doc{i}", 5) for i in range(7)}
    assignment = sd.assign_groups(
        groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=1
    )
    assert sorted(assignment) == sorted(groups)
    assert all(v in sd.SPLITS for v in assignment.values())


def test_line_count_balancing_beats_document_count():
    """One huge document plus many small ones. Balancing by line count must not
    put the huge document alone in val — that is what a naive 70/15/15 by
    document count does, and it is why the script bin-packs on line totals."""
    groups = {"huge": rows_for("huge", 600)}
    groups.update({f"small{i}": rows_for(f"small{i}", 30) for i in range(11)})
    assignment = sd.assign_groups(
        groups, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
    )
    lines = dict.fromkeys(sd.SPLITS, 0)
    for doc, split in assignment.items():
        lines[split] += len(groups[doc])
    total = sum(lines.values())
    assert assignment["huge"] == "train", "the largest document belongs in train"
    assert lines["train"] / total > 0.55


def test_deterministic_for_a_fixed_seed():
    groups = {f"doc{i}": rows_for(f"doc{i}", 10 + i) for i in range(9)}
    kwargs = {"ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}
    assert sd.assign_groups(groups, seed=7, **kwargs) == sd.assign_groups(
        groups, seed=7, **kwargs
    )


# ── Grouping ────────────────────────────────────────────────────────────────


def test_group_rows_groups_by_source_document_not_page():
    rows = rows_for("docA", 6, pages=3) + rows_for("docB", 4, pages=2)
    groups = sd.group_rows(rows, "source_document")
    assert set(groups) == {"docA", "docB"}
    assert len(groups["docA"]) == 6


def test_rows_missing_the_group_key_fall_back_and_are_not_dropped(capsys):
    rows = rows_for("docA", 3)
    for r in rows:
        del r["source_document"]
    groups = sd.group_rows(rows, "source_document")
    assert sum(len(v) for v in groups.values()) == 3
    assert "WARNING" in capsys.readouterr().out


# ── Leakage report ──────────────────────────────────────────────────────────


def test_leakage_report_is_clean_for_a_disjoint_split():
    splits = {
        "train": rows_for("d1", 10),
        "val": rows_for("d2", 3),
        "test": rows_for("d3", 3),
    }
    report = sd.build_leakage_report(splits, group_key="source_document")
    assert report["leakage_free"] is True
    for ov in report["overlaps"].values():
        assert ov["shared_groups"] == 0
        assert ov["shared_source_pages"] == 0
        assert ov["shared_image_paths"] == 0


def test_leakage_report_catches_a_shared_document():
    shared = rows_for("d1", 10)
    splits = {"train": shared[:7], "val": shared[7:], "test": rows_for("d2", 3)}
    report = sd.build_leakage_report(splits, group_key="source_document")
    assert report["leakage_free"] is False
    assert report["overlaps"]["train__val"]["shared_groups"] == 1
    assert "d1" in report["overlaps"]["train__val"]["shared_group_examples"]


def test_leakage_report_catches_a_shared_page_within_distinct_documents():
    """A page listed under two document ids is still the same pixels."""
    train = rows_for("d1", 4)
    val = rows_for("d2", 4)
    val[0]["source_image"] = train[0]["source_image"]
    report = sd.build_leakage_report(
        {"train": train, "val": val, "test": rows_for("d3", 3)},
        group_key="source_document",
    )
    assert report["leakage_free"] is False
    assert report["overlaps"]["train__val"]["shared_source_pages"] == 1


def test_duplicate_labels_are_reported_but_are_not_leakage():
    train = rows_for("d1", 5, text="سلام")
    val = rows_for("d2", 5, text="سلام")
    report = sd.build_leakage_report(
        {"train": train, "val": val, "test": rows_for("d3", 3)},
        group_key="source_document",
    )
    assert report["overlaps"]["train__val"]["duplicate_label_strings"] == 5
    assert report["leakage_free"] is True, (
        "identical text in two different documents is legitimate, not leakage"
    )


def test_empty_labels_are_not_counted_as_duplicates():
    """A `--backend none` crop manifest has every label empty; counting "" as a
    shared string would report phantom duplicates for every split pair."""
    train = rows_for("d1", 5)
    val = rows_for("d2", 5)
    for r in train + val:
        r["text"] = ""
    report = sd.build_leakage_report(
        {"train": train, "val": val, "test": rows_for("d3", 2)},
        group_key="source_document",
    )
    assert report["overlaps"]["train__val"]["duplicate_label_strings"] == 0
    assert report["unlabeled_rows"] == 10


# ── End-to-end through main() ───────────────────────────────────────────────


def test_main_writes_disjoint_splits(tmp_path, monkeypatch):
    import json

    inp = tmp_path / "lines_v2.jsonl"
    rows = []
    for i in range(12):
        rows += rows_for(f"doc{i}", 10 + i * 5)
    inp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )

    out = {name: tmp_path / f"{name}.jsonl" for name in ("train", "val", "test")}
    monkeypatch.setattr(sd, "TRAIN_OUT", out["train"])
    monkeypatch.setattr(sd, "VAL_OUT", out["val"])
    monkeypatch.setattr(sd, "TEST_OUT", out["test"])
    monkeypatch.setattr(sd, "LEAKAGE_REPORT", tmp_path / "leak.json")
    monkeypatch.setattr("sys.argv", ["split_dataset.py", "--input", str(inp)])

    assert sd.main() == 0

    docs = {}
    for name, path in out.items():
        docs[name] = {
            json.loads(l)["source_document"] for l in path.read_text(encoding="utf-8").splitlines() if l
        }
    assert docs["train"] & docs["val"] == set()
    assert docs["train"] & docs["test"] == set()
    assert docs["val"] & docs["test"] == set()

    total = sum(
        len([l for l in p.read_text(encoding="utf-8").splitlines() if l])
        for p in out.values()
    )
    assert total == len(rows), "every input row must land in exactly one split"

    report = json.loads((tmp_path / "leak.json").read_text(encoding="utf-8"))
    assert report["leakage_free"] is True


def test_main_refuses_when_there_are_too_few_groups(tmp_path, monkeypatch):
    import json

    inp = tmp_path / "tiny.jsonl"
    rows = rows_for("only", 5)
    inp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    monkeypatch.setattr("sys.argv", ["split_dataset.py", "--input", str(inp)])
    assert sd.main() == 1


def test_main_rejects_impossible_ratios(tmp_path, monkeypatch):
    inp = tmp_path / "x.jsonl"
    inp.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["split_dataset.py", "--input", str(inp), "--train-ratio", "0.9",
         "--val-ratio", "0.2"],
    )
    assert sd.main() == 1


@pytest.mark.parametrize("missing", ["nope.jsonl"])
def test_main_reports_a_missing_input(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(
        "sys.argv", ["split_dataset.py", "--input", str(tmp_path / missing)]
    )
    assert sd.main() == 1
