"""Queue ordering, blind switch, and covert repeat injection."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "new_gold" / "manifest.jsonl"

DEFAULT_QUEUE = "new_gold"
DEFAULT_SEED = 42
DEFAULT_BLIND_COUNT = 60
DEFAULT_REPEAT_RATE = 0.05
DEFAULT_PREFILL_MODEL = "qwen3-vl-2b-persian"


@dataclass(frozen=True)
class QueueItem:
    presentation_index: int
    line_id: str
    image_path: str
    crop_path: str
    source_document: str
    source_image: str
    blind_mode: bool
    is_repeat: bool
    prefill_model: str

    def to_dict(self) -> dict:
        return {
            "presentation_index": self.presentation_index,
            "line_id": self.line_id,
            "image_path": self.image_path,
            "crop_path": self.crop_path,
            "source_document": self.source_document,
            "source_image": self.source_image,
            "blind_mode": self.blind_mode,
            "is_repeat": self.is_repeat,  # server-only; strip before client if needed
            "prefill_model": self.prefill_model,
        }


def load_manifest(path: Path | None = None) -> list[dict]:
    p = Path(path or DEFAULT_MANIFEST)
    rows: list[dict] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_queue_order(
    manifest: list[dict],
    *,
    seed: int = DEFAULT_SEED,
    blind_count: int = DEFAULT_BLIND_COUNT,
    repeat_rate: float = DEFAULT_REPEAT_RATE,
    prefill_model: str = DEFAULT_PREFILL_MODEL,
) -> list[QueueItem]:
    """Seeded shuffle of unique lines, then inject ~repeat_rate covert repeats.

    Blind mode applies to the first ``blind_count`` unique lines in the shuffled
    order. Repeats inherit the blind/prefill flag of their first presentation.
    """
    if not manifest:
        return []
    if not 0.0 <= repeat_rate < 1.0:
        raise ValueError("repeat_rate must be in [0, 1)")
    blind_count = max(0, min(int(blind_count), len(manifest)))

    rng = random.Random(seed)
    unique = list(manifest)
    rng.shuffle(unique)

    blind_ids = {row["line_id"] for row in unique[:blind_count]}
    base: list[QueueItem] = []
    for i, row in enumerate(unique):
        lid = row["line_id"]
        base.append(
            QueueItem(
                presentation_index=i,
                line_id=lid,
                image_path=row.get("image_path") or row.get("crop_path") or "",
                crop_path=row.get("crop_path") or row.get("image_path") or "",
                source_document=row.get("source_document") or "",
                source_image=row.get("source_image") or "",
                blind_mode=lid in blind_ids,
                is_repeat=False,
                prefill_model=prefill_model,
            )
        )

    n_repeats = int(round(len(unique) * repeat_rate))
    if n_repeats <= 0 or len(base) < 2:
        # Re-index presentation_index densely
        return [
            QueueItem(
                presentation_index=i,
                line_id=it.line_id,
                image_path=it.image_path,
                crop_path=it.crop_path,
                source_document=it.source_document,
                source_image=it.source_image,
                blind_mode=it.blind_mode,
                is_repeat=it.is_repeat,
                prefill_model=it.prefill_model,
            )
            for i, it in enumerate(base)
        ]

    # Insert repeats after their first occurrence (never as the very first item).
    out = list(base)
    for _ in range(n_repeats):
        # Pick a source already in the queue (unique item)
        src = rng.choice(base)
        # Insert position strictly after first occurrence of that line_id
        first_idx = next(i for i, it in enumerate(out) if it.line_id == src.line_id)
        insert_at = rng.randint(first_idx + 1, len(out))
        out.insert(
            insert_at,
            QueueItem(
                presentation_index=-1,
                line_id=src.line_id,
                image_path=src.image_path,
                crop_path=src.crop_path,
                source_document=src.source_document,
                source_image=src.source_image,
                blind_mode=src.blind_mode,
                is_repeat=True,
                prefill_model=src.prefill_model,
            ),
        )

    return [
        QueueItem(
            presentation_index=i,
            line_id=it.line_id,
            image_path=it.image_path,
            crop_path=it.crop_path,
            source_document=it.source_document,
            source_image=it.source_image,
            blind_mode=it.blind_mode,
            is_repeat=it.is_repeat,
            prefill_model=it.prefill_model,
        )
        for i, it in enumerate(out)
    ]


def context_neighbors(
    order: list[QueueItem] | list[dict],
    index: int,
    *,
    window: int = 1,
) -> dict:
    def _get(it: QueueItem | dict, key: str):
        return it[key] if isinstance(it, dict) else getattr(it, key)

    prev_items = []
    next_items = []
    for off in range(1, window + 1):
        if index - off >= 0:
            it = order[index - off]
            prev_items.append(
                {
                    "presentation_index": _get(it, "presentation_index"),
                    "crop_path": _get(it, "crop_path"),
                    "line_id": _get(it, "line_id"),
                }
            )
        if index + off < len(order):
            it = order[index + off]
            next_items.append(
                {
                    "presentation_index": _get(it, "presentation_index"),
                    "crop_path": _get(it, "crop_path"),
                    "line_id": _get(it, "line_id"),
                }
            )
    prev_items.reverse()
    return {"prev": prev_items, "next": next_items}
