"""Tests for scripts/label_gold_interactive.py — the gold-set sampler.

The sampling logic is what makes the resulting CER trustworthy, so it is tested
directly: gold must come from the held-out split, spread across substantial
documents, and be reproducible so `--resume` compares against the same sample.
"""

import json

import pytest

from scripts.tests.conftest import load_script

lg = load_script("label_gold_under_test", "scripts/label_gold_interactive.py")


def rows_for(doc: str, n: int) -> list[dict]:
    return [
        {
            "image_path": f"dataset/lines_v3/{doc}_{i}.jpg",
            "text": "",
            "source_image": f"dataset/raw_images/{doc}_p{i // 20:03d}.png",
            "source_document": doc,
            "line_index": i,
            "bbox": [0, 0, 400, 60],
        }
        for i in range(n)
    ]


# ── Sampling: substantial documents only ────────────────────────────────────


def test_single_page_web_documents_are_excluded():
    """Test has 60 groups but 56 are single-page crawled images with 1-3 lines.
    Sampling evenly across all 60 would spend the typing budget on web
    screenshots instead of exam handwriting."""
    rows = rows_for("real_notebook", 600) + rows_for("real_exam", 300)
    for i in range(56):
        rows += rows_for(f"web/uuid-{i}", 2)
    picked = lg.sample_gold_todo(rows, target=240, seed=0)
    docs = {r["source_document"] for r in picked}
    assert docs == {"real_notebook", "real_exam"}


def test_sample_is_spread_evenly_across_substantial_documents():
    rows = rows_for("a", 600) + rows_for("b", 300) + rows_for("c", 120)
    picked = lg.sample_gold_todo(rows, target=240, seed=0)
    counts = {}
    for r in picked:
        counts[r["source_document"]] = counts.get(r["source_document"], 0) + 1
    assert counts == {"a": 80, "b": 80, "c": 80}


def test_target_is_respected():
    rows = rows_for("a", 500) + rows_for("b", 500)
    assert len(lg.sample_gold_todo(rows, target=100, seed=0)) == 100


def test_a_document_smaller_than_its_quota_is_not_oversampled():
    rows = rows_for("big", 500) + rows_for("small", 60)
    picked = lg.sample_gold_todo(rows, target=240, seed=0)
    small = [r for r in picked if r["source_document"] == "small"]
    assert len(small) == 60, "cannot sample more lines than the document has"
    assert len({r["image_path"] for r in small}) == 60, "and never the same line twice"


def test_falls_back_to_all_documents_when_none_are_substantial():
    """A tiny corpus must still yield a sample rather than nothing."""
    rows = rows_for("tiny_a", 10) + rows_for("tiny_b", 10)
    picked = lg.sample_gold_todo(rows, target=8, seed=0)
    assert picked
    assert {r["source_document"] for r in picked} == {"tiny_a", "tiny_b"}


def test_no_duplicate_crops_in_a_sample():
    rows = rows_for("a", 300) + rows_for("b", 300)
    picked = lg.sample_gold_todo(rows, target=240, seed=0)
    assert len({r["image_path"] for r in picked}) == len(picked)


# ── Reproducibility: --resume depends on it ─────────────────────────────────


def test_same_seed_gives_the_same_sample():
    rows = rows_for("a", 400) + rows_for("b", 400)
    first = [r["image_path"] for r in lg.sample_gold_todo(rows, target=100, seed=7)]
    second = [r["image_path"] for r in lg.sample_gold_todo(rows, target=100, seed=7)]
    assert first == second, "--resume compares against the sample, so it must be stable"


def test_different_seed_gives_a_different_sample():
    rows = rows_for("a", 400) + rows_for("b", 400)
    a = [r["image_path"] for r in lg.sample_gold_todo(rows, target=100, seed=1)]
    b = [r["image_path"] for r in lg.sample_gold_todo(rows, target=100, seed=2)]
    assert a != b


def test_documents_are_interleaved_not_grouped():
    """All of one document's lines in a row means fatigue lands on one writer."""
    rows = rows_for("a", 300) + rows_for("b", 300) + rows_for("c", 300)
    picked = lg.sample_gold_todo(rows, target=240, seed=0)
    docs = [r["source_document"] for r in picked[:30]]
    assert len(set(docs)) > 1


# ── Entries carry what the scorer needs ────────────────────────────────────


def test_sampled_rows_carry_the_fields_the_scorer_joins_on():
    rows = rows_for("a", 200)
    for r in lg.sample_gold_todo(rows, target=20, seed=0):
        # eval_gold_subset.py joins predictions on image_path and can group by
        # source_document; both must survive sampling.
        assert r["image_path"] and r["source_document"]
        assert "line_index" in r


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert lg.load_jsonl(path) == [{"a": 1}, {"a": 2}]


# ── End-to-end through main(), driven by scripted input ────────────────────


def _run(monkeypatch, capsys, tmp_path, keystrokes, argv_extra=(), rows=None):
    """Drive main() with scripted keystrokes against a throwaway corpus.

    `PROJECT_ROOT` is repointed at tmp_path and stub crop files are created there,
    because main() skips any row whose crop is missing from disk — without the
    files it would skip every row and test nothing.
    """
    rows = rows if rows is not None else rows_for("doc_a", 120) + rows_for("doc_b", 120)
    for row in rows:
        crop = tmp_path / row["image_path"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.touch()

    src = tmp_path / "test_v3.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    out = tmp_path / "gold.jsonl"
    keys = iter(keystrokes)
    monkeypatch.setattr("builtins.input", lambda *a: next(keys))
    monkeypatch.setattr(lg, "show", lambda *a, **k: None)
    monkeypatch.setattr(lg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["label_gold_interactive.py", "--source", str(src), "--out", str(out),
         "--no-open", *argv_extra],
    )
    rc = lg.main()
    return rc, out, capsys.readouterr().out


def test_main_appends_each_entry_as_it_is_typed(monkeypatch, capsys, tmp_path):
    rc, out, _ = _run(monkeypatch, capsys, tmp_path, ["سلام", "کتاب", "q"])
    assert rc == 0
    written = lg.load_jsonl(out)
    assert [r["text"] for r in written] == ["سلام", "کتاب"]
    assert all(r["image_path"].startswith("dataset/lines_v3/") for r in written)


def test_main_records_illegible_as_an_empty_label(monkeypatch, capsys, tmp_path):
    rc, out, _ = _run(monkeypatch, capsys, tmp_path, ["", "q"])
    assert rc == 0
    assert lg.load_jsonl(out)[0]["text"] == ""


def test_skip_records_nothing(monkeypatch, capsys, tmp_path):
    rc, out, _ = _run(monkeypatch, capsys, tmp_path, ["s", "سلام", "q"])
    written = lg.load_jsonl(out)
    assert len(written) == 1, "the skipped crop must not be recorded"
    assert written[0]["text"] == "سلام"


def test_undo_removes_the_previous_entry_from_the_file(monkeypatch, capsys, tmp_path):
    rc, out, _ = _run(monkeypatch, capsys, tmp_path, ["اشتباه", "u", "درست", "q"])
    written = lg.load_jsonl(out)
    assert [r["text"] for r in written] == ["درست"]


def test_resume_skips_already_labelled_crops(monkeypatch, capsys, tmp_path):
    _run(monkeypatch, capsys, tmp_path, ["اول", "q"])
    rc, out, printed = _run(monkeypatch, capsys, tmp_path, ["دوم", "q"])
    assert "already done: 1" in printed
    written = lg.load_jsonl(out)
    assert [r["text"] for r in written] == ["اول", "دوم"]
    assert len({r["image_path"] for r in written}) == 2


def test_no_resume_reoffers_labelled_crops(monkeypatch, capsys, tmp_path):
    _run(monkeypatch, capsys, tmp_path, ["اول", "q"])
    _, _, printed = _run(monkeypatch, capsys, tmp_path, ["q"], argv_extra=["--no-resume"])
    assert "already done" not in printed


def test_session_limit_stops_early(monkeypatch, capsys, tmp_path):
    rc, out, printed = _run(
        monkeypatch, capsys, tmp_path, ["a", "b", "c"], argv_extra=["--limit", "2"]
    )
    assert len(lg.load_jsonl(out)) == 2
    assert "Session limit reached" in printed


def test_artifact_input_is_challenged_and_can_be_rejected(monkeypatch, capsys, tmp_path):
    # REAL: "|---|---|---|" is a label the old pipeline produced 18 times.
    rc, out, printed = _run(monkeypatch, capsys, tmp_path, ["|---|---|---|", "n", "q"])
    assert "artifact" in printed
    assert not out.exists() or lg.load_jsonl(out) == []


def test_artifact_input_can_be_kept_when_confirmed(monkeypatch, capsys, tmp_path):
    rc, out, _ = _run(monkeypatch, capsys, tmp_path, ["الف | ب | ج", "y", "q"])
    assert lg.load_jsonl(out)[0]["text"] == "الف | ب | ج"


def test_interrupt_preserves_everything_already_typed(monkeypatch, capsys, tmp_path):
    def keys(*a):
        if not getattr(keys, "used", False):
            keys.used = True
            return "سلام"
        raise KeyboardInterrupt

    rows = rows_for("d", 120)
    for row in rows:
        crop = tmp_path / row["image_path"]
        crop.parent.mkdir(parents=True, exist_ok=True)
        crop.touch()

    src = tmp_path / "test_v3.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    out = tmp_path / "gold.jsonl"
    monkeypatch.setattr("builtins.input", keys)
    monkeypatch.setattr(lg, "show", lambda *a, **k: None)
    monkeypatch.setattr(lg, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["label_gold_interactive.py", "--source", str(src), "--out", str(out),
         "--no-open"],
    )
    assert lg.main() == 0
    assert [r["text"] for r in lg.load_jsonl(out)] == ["سلام"]


def test_a_missing_crop_is_reported_and_skipped(monkeypatch, capsys, tmp_path):
    """A manifest row whose crop is gone must not silently consume a keystroke."""
    rows = rows_for("d", 60)
    src = tmp_path / "test_v3.jsonl"
    src.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    out = tmp_path / "gold.jsonl"
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("should not prompt"))
    monkeypatch.setattr(lg, "show", lambda *a, **k: None)
    monkeypatch.setattr(lg, "PROJECT_ROOT", tmp_path)   # no crop files created
    monkeypatch.setattr(
        "sys.argv",
        ["label_gold_interactive.py", "--source", str(src), "--out", str(out),
         "--no-open"],
    )
    assert lg.main() == 0
    assert "MISSING" in capsys.readouterr().out


def test_main_reports_a_missing_source(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        ["label_gold_interactive.py", "--source", str(tmp_path / "nope.jsonl")],
    )
    assert lg.main() == 1


@pytest.mark.parametrize("keystrokes", [["q"], ["s", "q"]])
def test_quitting_immediately_is_not_an_error(monkeypatch, capsys, tmp_path, keystrokes):
    rc, _, _ = _run(monkeypatch, capsys, tmp_path, keystrokes)
    assert rc == 0
