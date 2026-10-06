"""Tests for deterministic page-level splitting."""

from eval.split import (
    assign_pages,
    build_split_manifests,
    _document_leakage_check,
    _leakage_check,
)


def test_no_page_in_two_splits():
    rows = []
    for page_i in range(20):
        page = f"dataset/raw_images/doc_p{page_i:03d}.png"
        for li in range(3):
            rows.append(
                {
                    "image_path": f"dataset/lines_v3/{page_i}_{li}.jpg",
                    "text": f"line {li}",
                    "source_document": "doc",
                    "line_index": li,
                    "source_image": page,
                }
            )
    manifests = build_split_manifests(rows, seed=42)
    leak = _leakage_check(manifests)
    assert leak["ok"] is True
    total_pages = sum(m["n_pages"] for m in manifests.values())
    assert total_pages == 20


def test_deterministic_same_seed():
    pages = [f"p{i}" for i in range(30)]
    a = assign_pages(pages, seed=7)
    b = assign_pages(pages, seed=7)
    assert a == b


def test_different_seed_can_differ():
    pages = [f"p{i}" for i in range(30)]
    a = assign_pages(pages, seed=1)
    b = assign_pages(pages, seed=2)
    assert a != b


def test_gold_test_policy_string(tmp_path):
    rows = [
        {
            "image_path": "x.jpg",
            "text": "سلام",
            "source_document": "d",
            "line_index": 0,
            "source_image": f"page_{i}.png",
        }
        for i in range(10)
    ]
    manifests = build_split_manifests(rows, seed=0)
    assert "never use for training" in manifests["gold_test"]["policy"].lower() or \
        "never" in manifests["gold_test"]["policy"]


def test_document_split_no_doc_leakage():
    rows = []
    for doc_i in range(6):
        for page_i in range(4):
            page = f"dataset/raw_images/doc{doc_i}_p{page_i}.png"
            rows.append(
                {
                    "image_path": f"dataset/lines_v3/{doc_i}_{page_i}.jpg",
                    "text": "سلام",
                    "source_document": f"doc{doc_i}",
                    "line_index": 0,
                    "source_image": page,
                }
            )
    manifests = build_split_manifests(rows, seed=42, split_by="document")
    assert _leakage_check(manifests)["ok"] is True
    assert _document_leakage_check(manifests)["ok"] is True
    assert manifests["train"]["split_by"] == "document"


def test_page_split_can_leak_documents():
    rows = []
    for page_i in range(20):
        page = f"dataset/raw_images/doc_p{page_i:03d}.png"
        rows.append(
            {
                "image_path": f"x{page_i}.jpg",
                "text": "سلام",
                "source_document": "same_doc",
                "line_index": 0,
                "source_image": page,
            }
        )
    manifests = build_split_manifests(rows, seed=42, split_by="page")
    leak = _document_leakage_check(manifests)
    assert leak["ok"] is False
    assert leak["n_leaked_documents"] == 1
