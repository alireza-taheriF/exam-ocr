"""Pure logic tests for queue ordering, repeats, and blind switch."""

from __future__ import annotations

from labeling.queue import build_queue_order


def _manifest(n: int = 20) -> list[dict]:
    return [
        {
            "line_id": f"L{i:03d}",
            "image_path": f"dataset/lines_v3/{i:03d}.jpg",
            "crop_path": f"dataset/lines_v3/{i:03d}.jpg",
            "source_document": f"doc{i % 3}",
            "source_image": f"p{i}.png",
        }
        for i in range(n)
    ]


def test_blind_first_n_unique():
    order = build_queue_order(_manifest(100), seed=1, blind_count=60, repeat_rate=0.0)
    assert len(order) == 100
    blinds = [it for it in order if it.blind_mode]
    assert len(blinds) == 60
    # First 60 unique in shuffled order are blind
    unique_order = []
    seen = set()
    for it in order:
        if it.line_id not in seen:
            unique_order.append(it)
            seen.add(it.line_id)
    assert all(it.blind_mode for it in unique_order[:60])
    assert all(not it.blind_mode for it in unique_order[60:])


def test_repeat_injection_rate_and_covert_flag():
    order = build_queue_order(_manifest(100), seed=7, blind_count=10, repeat_rate=0.05)
    assert len(order) == 105  # 100 + round(5)
    repeats = [it for it in order if it.is_repeat]
    assert len(repeats) == 5
    # Repeat inherits blind flag of first occurrence
    first_blind = {it.line_id: it.blind_mode for it in order if not it.is_repeat}
    for it in repeats:
        assert it.blind_mode == first_blind[it.line_id]
        # first occurrence appears before the repeat
        first_idx = next(i for i, x in enumerate(order) if x.line_id == it.line_id)
        assert first_idx < it.presentation_index


def test_seed_deterministic():
    a = [it.line_id for it in build_queue_order(_manifest(30), seed=42, repeat_rate=0.1)]
    b = [it.line_id for it in build_queue_order(_manifest(30), seed=42, repeat_rate=0.1)]
    c = [it.line_id for it in build_queue_order(_manifest(30), seed=43, repeat_rate=0.1)]
    assert a == b
    assert a != c


def test_configurable_blind_count():
    order = build_queue_order(_manifest(50), seed=0, blind_count=5, repeat_rate=0.0)
    assert sum(1 for it in order if it.blind_mode) == 5
