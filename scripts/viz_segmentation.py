#!/usr/bin/env python3
"""Draw pipeline/segment.py boxes on real pages and compare with the Surya boxes.

For each page: an overlay JPEG (OpenCV text lines red, image blocks blue, Surya
reference lines from lines_v3.jsonl green, thin) and a summary row. The Surya
boxes are those that survived the geometry filter, so they are a reference,
not ground truth.

    .venv-qwen/bin/python scripts/viz_segmentation.py --out dataset/reports/segmentation \
        dataset/raw_images/<page>.png [...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from pipeline.segment import _segment_opencv  # noqa: E402


def iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def inside(a: list[int], b: list[int]) -> float:
    """Fraction of box a's area that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lines", type=Path, default=PROJECT_ROOT / "dataset" / "lines_v3.jsonl")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    surya: dict[str, list[list[int]]] = defaultdict(list)
    for line in args.lines.open(encoding="utf-8"):
        r = json.loads(line)
        surya[r["source_image"]].append(r["bbox"])

    summary = []
    for path in args.images:
        rel = str(path.resolve().relative_to(PROJECT_ROOT))
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        items = _segment_opencv(rgb)
        text = [it["bbox"] for it in items if it["kind"] == "text"]
        figs = [it["bbox"] for it in items if it["kind"] == "image"]
        ref = surya.get(rel, [])

        thick = max(2, bgr.shape[1] // 600)
        for b in ref:
            cv2.rectangle(bgr, tuple(b[:2]), tuple(b[2:]), (0, 170, 0), max(1, thick // 2))
        for k, b in enumerate(text, 1):
            cv2.rectangle(bgr, tuple(b[:2]), tuple(b[2:]), (0, 0, 230), thick)
            cv2.putText(bgr, str(k), (b[2] + 4, b[3]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * thick, (0, 0, 230), max(1, thick // 2))
        for b in figs:
            cv2.rectangle(bgr, tuple(b[:2]), tuple(b[2:]), (230, 90, 0), thick)
        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # A Surya line is "matched" if some OpenCV box overlaps it with IoU ≥ 0.5.
        matched = sum(1 for s in ref if any(iou(s, t) >= 0.5 for t in text))
        # Looser, Surya often splits one written line into several boxes:
        # covered = ≥70% of a Surya box inside one OpenCV box that is not a
        # multi-line merge (≤2× the Surya box height).
        covered = sum(1 for s in ref if any(
            inside(s, t) >= 0.7 and (t[3] - t[1]) <= 2 * (s[3] - s[1]) for t in text))
        # An OpenCV box is "on text" if it overlaps some Surya box at all
        # (spiral rings, table edges and ruling leftovers are not).
        on_text = sum(1 for t in text if any(inside(s, t) > 0.3 for s in ref))
        heights = [b[3] - b[1] for b in text]
        row = {
            "page": rel,
            "opencv_text_lines": len(text),
            "opencv_image_blocks": len(figs),
            "surya_ref_lines": len(ref),
            "surya_matched_iou50": matched,
            "surya_recall_iou50": round(matched / len(ref), 3) if ref else None,
            "surya_covered": round(covered / len(ref), 3) if ref else None,
            "opencv_on_text": round(on_text / len(text), 3) if text else None,
            "median_line_height": statistics.median(heights) if heights else None,
            "tall_lines_gt2x_median": sum(h > 2 * statistics.median(heights) for h in heights) if heights else 0,
        }
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    def mean(key: str) -> float | None:
        vals = [r[key] for r in summary if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    totals = {k: mean(k) for k in ("surya_recall_iou50", "surya_covered", "opencv_on_text")}
    print("MEAN", json.dumps(totals))
    (args.out / "summary.json").write_text(
        json.dumps({"mean": totals, "pages": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
