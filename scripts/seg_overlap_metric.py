#!/usr/bin/env python3
"""Reference-free segmentation check: written lines on a page must not overlap.

For each page prints the number of OpenCV text boxes and how many of them share
a notable area with another box (overlap ≥ 20% of the smaller one), plus the
count of boxes taller than 2.2× the median. Lower is better; a clean page has 0.

    .venv/bin/python scripts/seg_overlap_metric.py [pages...]
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.segment import segment_page  # noqa: E402  (backend chosen by env SEGMENTER)

DEFAULT_PAGES = [
    "CamScanner_۲۰۲۵-۰۶-۰۳_۲۲_۳۲_p002", "CamScanner_۲۰۲۵-۰۶-۰۳_۲۲_۳۲_p005",
    "Summary_of_Yule_p034", "اصول_تنظیم_و_کنترل_بودجه_دولت_p001", "تحلیل_سازه_مهندس_طاحونی_p093",
    "تست_های_ا_مار_کاربردی1_فصل_1تا4هادی_رنجبران_p014", "جزوه_دست_نویس_اصول_حسابداری_1_p025",
    "جزوه_دست_نویس_حسابداری_بخش_عمومی_p023", "جزوه_دست_نویس_حسابداری_پیشرفته_1_p020",
]


def overlap_frac(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller else 0.0


def measure(boxes: list[list[int]]) -> dict:
    hit = {i for i, a in enumerate(boxes) for j, b in enumerate(boxes) if i != j and overlap_frac(a, b) >= 0.2}
    med = statistics.median(b[3] - b[1] for b in boxes) if boxes else 0
    return {"boxes": len(boxes), "overlapping": len(hit),
            "tall": sum(1 for b in boxes if b[3] - b[1] > 2.2 * med)}


def main() -> int:
    pages = [Path(p) for p in sys.argv[1:]] or [PROJECT_ROOT / "dataset" / "raw_images" / f"{n}.png" for n in DEFAULT_PAGES]
    tot = {"boxes": 0, "overlapping": 0, "tall": 0}
    for p in pages:
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = measure([it["bbox"] for it in segment_page(rgb) if it["kind"] == "text"])
        for k in tot:
            tot[k] += m[k]
        print(f"{p.stem[:46]:46s} boxes {m['boxes']:3d}  overlapping {m['overlapping']:3d}  tall {m['tall']:2d}")
    print(f"TOTAL boxes {tot['boxes']}  overlapping {tot['overlapping']} ({100*tot['overlapping']/max(1,tot['boxes']):.0f}%)  tall {tot['tall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
