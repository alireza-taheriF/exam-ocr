from labeling.db import append_correction, connect, latest_non_skip, undo_last


def test_append_only_undo(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    append_correction(
        conn,
        line_id="L1",
        queue="new_gold",
        model_text="m",
        corrected_text="one",
        action="edited",
        region_type="body",
        reviewer="t",
        started_at=None,
        seconds_spent=1.0,
        notes="",
        blind_mode=True,
        presentation_index=0,
        is_repeat=False,
        prefill_model="qwen3-vl-2b-persian",
        jsonl_path=tmp_path / "c.jsonl",
    )
    assert latest_non_skip(conn, "new_gold")["L1"]["corrected_text"] == "one"
    undo_last(conn, "new_gold")
    assert "L1" not in latest_non_skip(conn, "new_gold")
