"""Tests for scripts/extract_and_align_lines.py.

Focused on the two properties Phase 1 exists to guarantee:

* the geometry filter (Step 2) drops exactly what it claims to, and the report
  it writes adds up;
* a crop's identity is derived from the page and box it came from, so labels
  cannot drift onto the wrong image the way the legacy index-zip did.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.tests.conftest import load_script

ex = load_script("extract_under_test", "scripts/extract_and_align_lines.py")


# ── Step 2: geometry filter ─────────────────────────────────────────────────


def test_accepts_a_normal_text_line():
    stats = ex.GeometryStats()
    assert ex.passes_geometry(10, 10, 410, 60, stats) is True
    assert (stats.kept, stats.detected) == (1, 1)


@pytest.mark.parametrize(
    "box,attr",
    [
        ((0, 0, 400, 8), "drop_short"),        # 8px tall — the measured minimum
        ((0, 0, 400, 23), "drop_short"),       # just under the 24px floor
        ((0, 0, 40, 200), "drop_portrait"),    # portrait: cannot be a text line
        ((0, 0, 4000, 20), "drop_short"),      # 200:1 page rule, also too short
        ((0, 0, 800, 40), "drop_wide"),        # 20:1 aspect, tall enough
        ((0, 0, 0, 0), "drop_degenerate"),
        ((10, 10, 10, 60), "drop_degenerate"),  # zero width
    ],
)
def test_rejects_bad_geometry(box, attr):
    stats = ex.GeometryStats()
    assert ex.passes_geometry(*box, stats) is False
    assert getattr(stats, attr) == 1
    assert stats.kept == 0


def test_boundary_values_are_inclusive_where_documented():
    stats = ex.GeometryStats()
    # Exactly the 24px floor is kept (width chosen to stay under the aspect cap).
    assert ex.passes_geometry(0, 0, 300, 24, stats) is True
    assert ex.passes_geometry(0, 0, 300, 23, stats) is False
    assert stats.drop_short == 1
    # Exactly 16:1 is kept; one pixel wider is not.
    assert ex.passes_geometry(0, 0, 16 * 30, 30, stats) is True
    assert ex.passes_geometry(0, 0, 16 * 30 + 1, 30, stats) is False
    assert stats.drop_wide == 1


def test_square_crop_is_kept_but_one_pixel_taller_is_not():
    stats = ex.GeometryStats()
    assert ex.passes_geometry(0, 0, 40, 40, stats) is True
    assert ex.passes_geometry(0, 0, 40, 41, stats) is False
    assert stats.drop_portrait == 1


def test_thresholds_are_configurable():
    stats = ex.GeometryStats(min_height=8, max_aspect=250.0)
    assert ex.passes_geometry(0, 0, 400, 10, stats) is True
    assert ex.passes_geometry(0, 0, 2000, 10, stats) is True


def test_report_counts_reconcile():
    stats = ex.GeometryStats()
    for box in [(0, 0, 400, 50), (0, 0, 400, 8), (0, 0, 40, 200), (0, 0, 900, 40)]:
        ex.passes_geometry(*box, stats)
    report = stats.as_report()
    assert report["detected_boxes"] == 4
    assert report["kept"] == 1
    assert sum(report["dropped"].values()) + report["kept"] == report["detected_boxes"]
    assert report["kept_pct"] == 25.0


def test_report_is_json_serializable_and_records_thresholds():
    stats = ex.GeometryStats(min_height=30, max_aspect=12.0)
    ex.passes_geometry(0, 0, 400, 50, stats)
    stats.note_page_class(3508, 1, 1)
    report = stats.as_report()
    round_tripped = json.loads(json.dumps(report, ensure_ascii=False))
    assert round_tripped["thresholds"]["min_height_px"] == 30
    assert round_tripped["thresholds"]["max_aspect_ratio"] == 12.0
    assert round_tripped["thresholds"]["reject_portrait"] is True


def test_page_resolution_classes_are_tracked_separately():
    stats = ex.GeometryStats()
    stats.note_page_class(3508, 40, 34)     # 300-DPI PDF render
    stats.note_page_class(700, 30, 6)       # crawled web image
    stats.note_page_class(560, 20, 4)
    report = stats.as_report()["by_page_resolution"]
    assert report["high_res"]["pages"] == 1
    assert report["low_res"]["pages"] == 2
    assert report["low_res"]["detected"] == 50
    assert report["high_res"]["kept_pct"] > report["low_res"]["kept_pct"]


# ── Document identity (Step 4 depends on this) ──────────────────────────────


def test_pdf_pages_of_one_document_share_a_document_id():
    a = ex.derive_source_document("dataset/raw_images/جزوه_حسابداری_p018.png")
    b = ex.derive_source_document("dataset/raw_images/جزوه_حسابداری_p147.png")
    assert a == b == "جزوه_حسابداری"


def test_page_suffix_stripping_is_anchored_to_the_end():
    assert ex.derive_source_document("x/report_p12_final.png") == "report_p12_final"


def test_each_crawled_uuid_page_is_its_own_document():
    a = ex.derive_source_document(
        "dataset/raw_images/01d69564-3206-4196-9331-fdc323642cc0.webp"
    )
    b = ex.derive_source_document(
        "dataset/raw_images/6ffd323e-9a27-48b2-8cda-733e6982ab6f.webp"
    )
    assert a != b
    assert a.startswith("web/")


def test_document_id_is_independent_of_directory_and_extension():
    assert ex.derive_source_document("a/b/doc_p001.png") == ex.derive_source_document(
        "z/doc_p999.jpg"
    )


# ── Crop identity: content-addressed, so re-runs are idempotent ─────────────


def test_crop_name_is_stable_for_the_same_page_and_box():
    assert ex.crop_name("dataset/raw_images/p.png", [1, 2, 3, 4]) == ex.crop_name(
        "dataset/raw_images/p.png", [1, 2, 3, 4]
    )


def test_crop_name_differs_per_box_and_per_page():
    base = ex.crop_name("p.png", [0, 0, 100, 40])
    assert base != ex.crop_name("p.png", [0, 1, 100, 40])
    assert base != ex.crop_name("q.png", [0, 0, 100, 40])


def test_crop_name_is_a_jpg_filename():
    name = ex.crop_name("p.png", [0, 0, 10, 10])
    assert name.endswith(".jpg")
    assert len(name) == len("0123456789ab.jpg")


# ── Crop padding (Phase 1B: Surya boxes clip descenders) ────────────────────


def test_padding_grows_the_box_vertically_only():
    x1, y1, x2, y2 = ex.pad_bbox((100, 1000, 900, 1100), page_w=2000, page_h=3000)
    assert (x1, x2) == (100, 900), "horizontal extent must not change"
    assert y1 == 1000 - 8 and y2 == 1100 + 8   # 7.5% of h=100, rounded


def test_padding_is_proportional_to_box_height():
    _, y1, _, y2 = ex.pad_bbox((0, 500, 400, 700), page_w=1000, page_h=2000)
    assert (y1, y2) == (500 - 15, 700 + 15)    # 7.5% of h=200


def test_padding_clamps_at_the_top_of_the_page():
    _, y1, _, y2 = ex.pad_bbox((0, 3, 400, 103), page_w=1000, page_h=2000)
    assert y1 == 0
    assert y2 == 111


def test_padding_clamps_at_the_bottom_of_the_page():
    page_h = 3000
    _, y1, _, y2 = ex.pad_bbox((0, 2900, 400, 2998), page_w=1000, page_h=page_h)
    assert y2 == page_h
    assert y1 == 2900 - 7


def test_zero_padding_reproduces_the_detected_box():
    box = (10, 20, 30, 60)
    assert ex.pad_bbox(box, page_w=100, page_h=100, fraction=0.0) == box


def test_padded_crop_is_a_superset_of_the_detected_box():
    box = (50, 400, 650, 480)
    padded = ex.pad_bbox(box, page_w=1000, page_h=1000)
    assert padded[0] <= box[0] and padded[1] <= box[1]
    assert padded[2] >= box[2] and padded[3] >= box[3]


def test_padding_does_not_change_which_boxes_pass_the_geometry_filter():
    """The filter judges the detected box; padding applies to the crop only. A
    24px box padded to 27px must still be evaluated as 24px, or the filter's
    measured drop counts stop meaning anything."""
    stats = ex.GeometryStats()
    assert ex.passes_geometry(0, 0, 300, 23, stats) is False
    padded = ex.pad_bbox((0, 100, 300, 123), page_w=1000, page_h=1000)
    assert padded[3] - padded[1] > 23, "the crop really is taller"
    assert stats.drop_short == 1, "but the filter still saw 23px"


def test_crop_name_changes_when_padding_changes():
    """Crop files are content-addressed on the box actually written, so a v3
    padded crop cannot collide with the v2 unpadded crop of the same line."""
    box = (0, 100, 400, 160)
    unpadded = ex.crop_name("p.png", list(box))
    padded = ex.crop_name("p.png", list(ex.pad_bbox(box, page_w=1000, page_h=1000)))
    assert unpadded != padded


# ── Page deduplication ──────────────────────────────────────────────────────


def _write_png(path: Path, colour: int, size=(60, 40)) -> Path:
    Image.fromarray(np.full((size[1], size[0], 3), colour, dtype=np.uint8)).save(path)
    return path


def test_byte_identical_pages_are_deduplicated(tmp_path):
    a = _write_png(tmp_path / "a.png", 10)
    b = _write_png(tmp_path / "b.png", 10)   # identical pixels, different name
    c = _write_png(tmp_path / "c.png", 200)
    kept, info = ex.dedupe_pages([a, b, c])
    assert info["duplicate_pages_dropped"] == 1
    assert info["pages_in"] == 3 and info["pages_unique"] == 2
    assert len(kept) == 2
    assert a in kept and b not in kept, "the first path by name is the one kept"


def test_dedup_keeps_distinct_pages_untouched(tmp_path):
    pages = [_write_png(tmp_path / f"p{i}.png", i * 20) for i in range(4)]
    kept, info = ex.dedupe_pages(pages)
    assert info["duplicate_pages_dropped"] == 0
    assert kept == sorted(pages)


def test_dedup_records_which_page_replaced_which(tmp_path):
    a = _write_png(tmp_path / "a.png", 7)
    b = _write_png(tmp_path / "b.png", 7)
    _, info = ex.dedupe_pages([a, b])
    assert info["examples"][0]["dropped"].endswith("b.png")
    assert info["examples"][0]["kept"].endswith("a.png")


# ── Label validation at ingestion (Step 3 wired into Step 1) ────────────────


def test_validate_rejects_empty_and_illegible():
    assert ex._validate("") == "empty_transcription"
    assert ex._validate("   ") == "empty_transcription"
    assert ex._validate(ex.ILLEGIBLE_MARKER) == "model_reported_illegible"


def test_validate_rejects_the_real_markdown_table_artifact():
    # REAL: 18 rows of dataset/train.jsonl carry this as their entire label.
    assert ex._validate("|---|---|---|").startswith("artifact_")


def test_validate_accepts_real_persian_lines():
    for text in ["سلام دنیا", "موجودی کالای شعبه ۸۵۰/۰۰۰", "الف) ۲۵٪"]:
        assert ex._validate(text) is None


def test_validate_rejects_emoji_and_circled_digits():
    assert ex._validate("🎓 دانشگاه").startswith("artifact_")
    assert ex._validate("① اول").startswith("artifact_")


# ── Cost guardrail ──────────────────────────────────────────────────────────


def test_cost_estimate_scales_with_crop_area_and_count():
    small = ex.estimate_cost([(0, 0, 100, 40)] * 10)
    large = ex.estimate_cost([(0, 0, 1000, 100)] * 10)
    assert small["calls"] == large["calls"] == 10
    assert large["image_tokens"] > small["image_tokens"]
    assert large["by_model"]["claude-opus-4-8"]["standard_usd"] >= (
        small["by_model"]["claude-opus-4-8"]["standard_usd"]
    )


def test_batch_pricing_is_half_of_standard():
    est = ex.estimate_cost([(0, 0, 400, 50)] * 1000)
    for model in est["by_model"].values():
        assert model["batch_api_usd"] == pytest.approx(
            model["standard_usd"] / 2, abs=0.01
        )


def test_cheaper_model_costs_less():
    est = ex.estimate_cost([(0, 0, 400, 50)] * 5000)
    opus = est["by_model"]["claude-opus-4-8"]["standard_usd"]
    haiku = est["by_model"]["claude-haiku-4-5"]["standard_usd"]
    assert haiku < opus


def test_empty_plan_costs_nothing():
    est = ex.estimate_cost([])
    assert est["calls"] == 0
    assert est["by_model"]["claude-opus-4-8"]["standard_usd"] == 0.0
