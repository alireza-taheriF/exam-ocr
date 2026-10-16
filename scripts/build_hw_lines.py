#!/usr/bin/env python3
"""Build line crops from MR3z4/persian-handwriting-ocr for LoRA training.

The dataset ships page scans plus one CSV row per *word* (``page_id, block,
line, x, y, x2, y2, text``); it has no image column, so the Colab loader cannot
stream it. This script groups words into lines by ``(page_id, block, line)``,
joins them in file order (the annotators' reading order, checked by eye on
2026-10-13), crops the union box with the same vertical padding as
``lines_v3`` (7.5% per side) and writes manifests in the
``dataset/train_v3.jsonl`` shape.

Download first (≈0.7 GB):

    .venv/bin/python -c "from huggingface_hub import snapshot_download; \\
      snapshot_download('MR3z4/persian-handwriting-ocr', repo_type='dataset', \\
      local_dir='dataset/external/persian-handwriting-ocr', \\
      allow_patterns=['data/*.csv', 'raw/pages/*', 'LICENSE', 'README.md'])"
    .venv/bin/python scripts/build_hw_lines.py

Output under ``dataset/external/hw_lines/``: ``{train,validation,test}.jsonl``
and ``crops/``. The dataset's page-level splits are kept: ``train`` and
``validation`` may be trained on, ``test`` is an extra handwriting evaluation
set and must not be. Labels are the dataset's own and are known to be noisy
(its README asks for a human revision pass); the filters below only drop lines
whose geometry or text is clearly unusable.

License: CC-BY-NC-SA-4.0 (non-commercial). Check before shipping a model
trained on it in a paid product.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import normalize_text, vlm_artifact_reason  # noqa: E402

DEFAULT_SRC = PROJECT_ROOT / "dataset" / "external" / "persian-handwriting-ocr"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "external" / "hw_lines"
SPLITS = ("train", "validation", "test")
V_PAD = 0.075            # per side, as a fraction of line height (lines_v3)
H_PAD = 0.25             # per side, as a fraction of line height
MAX_CHARS = 150
MAX_HEIGHT_RATIO = 2.2   # union height / median word height; above = words from two lines


def group_lines(rows: list[dict]) -> dict[tuple[str, int, int], list[dict]]:
    lines: dict[tuple[str, int, int], list[dict]] = collections.OrderedDict()
    for r in rows:
        lines.setdefault((r["page_id"], int(r["block"]), int(r["line"])), []).append(r)
    return lines


def line_box(words: list[dict], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Union of word boxes, padded and clamped to the page."""
    w, h = size
    x1 = min(int(r["x"]) for r in words)
    y1 = min(int(r["y"]) for r in words)
    x2 = max(int(r["x2"]) for r in words)
    y2 = max(int(r["y2"]) for r in words)
    height = max(1, y2 - y1)
    vp, hp = int(round(V_PAD * height)), int(round(H_PAD * height))
    return max(0, x1 - hp), max(0, y1 - vp), min(w, x2 + hp), min(h, y2 + vp)


def reject_reason(words: list[dict]) -> str | None:
    text = " ".join(r["text"].strip() for r in words if r["text"].strip())
    if not normalize_text(text):
        return "empty"
    if len(text) > MAX_CHARS:
        return "too_long"
    if "\\" in text or "$" in text:
        return "latex"
    if vlm_artifact_reason(text):
        return "artifact"
    heights = [int(r["y2"]) - int(r["y"]) for r in words]
    if min(heights) <= 0 or any(int(r["x2"]) <= int(r["x"]) for r in words):
        return "bad_box"
    union = max(int(r["y2"]) for r in words) - min(int(r["y"]) for r in words)
    if union > MAX_HEIGHT_RATIO * statistics.median(heights):
        return "multi_line_group"
    return None


def build(src: Path, out: Path, *, limit_pages: int = 0) -> dict:
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    summary: dict = {"source": "MR3z4/persian-handwriting-ocr", "license": "CC-BY-NC-SA-4.0"}
    for split in SPLITS:
        rows = list(csv.DictReader((src / "data" / f"{split}.csv").open(encoding="utf-8")))
        lines = group_lines(rows)
        pages = sorted({k[0] for k in lines})
        if limit_pages:
            keep = set(pages[:limit_pages])
            lines = {k: v for k, v in lines.items() if k[0] in keep}
        counts: collections.Counter[str] = collections.Counter()
        by_page: dict[str, list[tuple[tuple[str, int, int], list[dict]]]] = collections.defaultdict(list)
        for key, words in lines.items():
            by_page[key[0]].append((key, words))
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as sink:
            for page_id, items in by_page.items():
                page = Image.open(src / items[0][1][0]["file_name"]).convert("RGB")
                for (pid, block, line), words in items:
                    reason = reject_reason(words)
                    if reason:
                        counts[reason] += 1
                        continue
                    x1, y1, x2, y2 = line_box(words, page.size)
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        counts["tiny"] += 1
                        continue
                    path = crops / f"{pid}_b{block:03d}_l{line:03d}.jpg"
                    page.crop((x1, y1, x2, y2)).save(path, quality=92)
                    sink.write(json.dumps({
                        "image_path": str(path.relative_to(PROJECT_ROOT)),
                        "text": " ".join(r["text"].strip() for r in words if r["text"].strip()),
                        "source_document": f"mr3z4/{pid}",
                        "line_index": line,
                        "block": block,
                        "label_backend": "mr3z4_gt",
                    }, ensure_ascii=False) + "\n")
                    counts["kept"] += 1
        summary[split] = {"pages": len(by_page), **dict(counts)}
        print(split, summary[split], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit-pages", type=int, default=0, help="per split, for a quick look")
    args = p.parse_args()
    if not (args.src / "data" / "train.csv").exists():
        print(f"dataset not found under {args.src}; see the download command in this file's docstring")
        return 1
    build(args.src, args.out, limit_pages=args.limit_pages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
