#!/usr/bin/env python3
"""Stack todo crops into numbered contact sheets so they can be read in a session.

Each sheet holds --per crops, one per row, with its index drawn in a left
margin and a thin rule between rows. A sidecar .json maps index -> image_path.
Rows already present in --done are skipped, so labeling is resumable.

    .venv/bin/python scripts/make_label_sheets.py --todo dataset/dev_todo.jsonl \
        --done dataset/dev_claude.jsonl --outdir /tmp/sheets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
MARGIN, MAX_W, MAX_H, MIN_H = 70, 1100, 220, 40  # overridable via --max-h/--min-h


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


FULLRES = False


def fit(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(MAX_W / w, MAX_H / h)
    if h * s < MIN_H:  # very flat crops: enlarge so the strokes are visible
        s = min(MIN_H / h, MAX_W / w)
    if FULLRES:  # never shrink below native size unless wider than the sheet
        s = max(s, min(1.0, MAX_W / w))
        if h * s < MIN_H:
            s = min(MIN_H / h, MAX_W / w)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def sheet(rows: list[dict], first_index: int) -> Image.Image:
    tiles = [fit(Image.open(ROOT / r["image_path"]).convert("RGB")) for r in rows]
    height = sum(t.height + 12 for t in tiles)
    out = Image.new("RGB", (MARGIN + MAX_W + 10, height), "white")
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    y = 0
    for i, t in enumerate(tiles):
        d.text((6, y + max(0, t.height // 2 - 16)), str(first_index + i), fill="red", font=font)
        out.paste(t, (MARGIN, y + 4))
        y += t.height + 12
        d.line([(0, y - 3), (out.width, y - 3)], fill=(150, 150, 150), width=1)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--todo", type=Path, required=True)
    p.add_argument("--done", type=Path, default=None)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--per", type=int, default=10)
    p.add_argument("--limit-sheets", type=int, default=0)
    p.add_argument("--fullres", action="store_true",
                   help="native resolution, 2x for flat crops (use with --per 4)")
    a = p.parse_args()
    global FULLRES, MAX_H, MIN_H
    if a.fullres:
        FULLRES, MAX_H, MIN_H = True, 600, 96

    done = {r["image_path"] for r in load(a.done)} if a.done else set()
    todo = [r for r in load(a.todo) if r["image_path"] not in done]
    a.outdir.mkdir(parents=True, exist_ok=True)
    for old in a.outdir.glob("sheet_*"):
        old.unlink()
    n = 0
    for k in range(0, len(todo), a.per):
        chunk = todo[k:k + a.per]
        name = f"sheet_{k // a.per:03d}"
        sheet(chunk, 1).save(a.outdir / f"{name}.png")
        (a.outdir / f"{name}.json").write_text(
            json.dumps({str(i + 1): r["image_path"] for i, r in enumerate(chunk)}, ensure_ascii=False))
        n += 1
        if a.limit_sheets and n >= a.limit_sheets:
            break
    print(f"{len(todo)} todo -> {n} sheets in {a.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
