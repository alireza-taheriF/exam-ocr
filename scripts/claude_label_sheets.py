#!/usr/bin/env python3
"""Label line crops by having Claude (in a Claude Code session) read them.

Crops are packed into numbered contact sheets so one image read covers ~10
lines. The session writes one transcription file per sheet; `ingest` turns
those into a labels JSONL that `scripts/eval_gate.py --engine labels` can score
and the LoRA data builder can train on.

    # 1. sheets (never shows existing labels, so gold can be read blind)
    .venv/bin/python scripts/claude_label_sheets.py make dataset/gold.jsonl outputs/claude_label/gold
    # 2. Claude reads outputs/claude_label/gold/sheet_000.png and writes sheet_000.txt:
    #        3<TAB>متن خط سوم
    #        4<TAB>∅            (illegible / not a single text line → empty label)
    # 3. labels
    .venv/bin/python scripts/claude_label_sheets.py ingest outputs/claude_label/gold dataset/claude_labels/gold.jsonl

Sealed crops (training/sealed.py) are refused unless --allow-sealed is given;
reading gold blind to measure the labeler is the one legitimate use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ILLEGIBLE = "∅"
SHEET_WIDTH = 1500      # stays under the image reader's downscale threshold
SHEET_MAX_HEIGHT = 1500
MARGIN = 70             # left gutter for the line number
GAP = 14
MAX_CROP_HEIGHT = 220   # page-sized "lines" are shrunk, not skipped
MAX_UPSCALE = 3.0


HALF = SHEET_WIDTH // 2  # short crops go two to a row


def _fit(im: Image.Image, room: int) -> Image.Image:
    scale = min(room / im.width, MAX_CROP_HEIGHT / im.height, MAX_UPSCALE)
    return im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)


def _fit_any(im: Image.Image) -> tuple[Image.Image, bool]:
    """Scale for a half-width slot if that loses no size, else for a full row."""
    half, full = _fit(im, HALF - MARGIN - 10), _fit(im, SHEET_WIDTH - MARGIN - 10)
    return (half, True) if half.size == full.size else (full, False)


def _font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make(manifest: Path, out_dir: Path, allow_sealed: bool, limit: int) -> None:
    rows = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()]
    if not allow_sealed:
        from training.sealed import sealed_crop_names
        sealed = sealed_crop_names()
        hit = [r for r in rows if Path(r["image_path"]).name in sealed]
        if hit:
            sys.exit(f"{len(hit)} sealed crops in {manifest}; pass --allow-sealed only to measure the labeler")
    if limit:
        rows = rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(30)

    index, rows_, height, n_sheets = [], [], 0, 0   # rows_: [[(num, im, half)], ...]

    def flush():
        nonlocal rows_, height, n_sheets
        if not rows_:
            return
        canvas = Image.new("RGB", (SHEET_WIDTH, height + GAP), "white")
        draw = ImageDraw.Draw(canvas)
        y = GAP
        for row in rows_:
            row_h = max(im.height for _, im, _ in row)
            for slot, (num, im, _) in enumerate(row):
                x0 = slot * HALF
                canvas.paste(im, (x0 + MARGIN, y))
                draw.text((x0 + 8, y + max(0, im.height // 2 - 16)), str(num), fill=(200, 0, 0), font=font)
            if len(row) == 2:
                draw.line([(HALF, y), (HALF, y + row_h)], fill=(150, 150, 255), width=2)
            y += row_h + GAP
            draw.line([(0, y - GAP // 2), (SHEET_WIDTH, y - GAP // 2)], fill=(150, 150, 255), width=2)
        canvas.save(out_dir / f"sheet_{n_sheets:03d}.png")
        n_sheets += 1
        rows_, height = [], 0

    for num, r in enumerate(rows):
        im, half = _fit_any(Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB"))
        last = rows_[-1] if rows_ else None
        if half and last and len(last) == 1 and last[0][2]:
            old_h = max(i.height for _, i, _ in last)
            if im.height > old_h and height + im.height - old_h > SHEET_MAX_HEIGHT:
                flush()
                rows_.append([(num, im, half)])
                height += im.height + GAP
            else:
                last.append((num, im, half))
                height += max(0, im.height - old_h)
        else:
            if rows_ and height + im.height + GAP > SHEET_MAX_HEIGHT:
                flush()
            rows_.append([(num, im, half)])
            height += im.height + GAP
        index.append({"num": num, "sheet": n_sheets, "image_path": r["image_path"],
                      "source_document": r.get("source_document"), "line_index": r.get("line_index"),
                      "source_image": r.get("source_image"), "line_id": r.get("line_id")})
    flush()
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} crops → {n_sheets} sheets in {out_dir}")


def read_transcriptions(sheet_dir: Path) -> dict[int, str]:
    """{crop number: text} from sheet_*.txt; "" means the reader marked it unreadable."""
    texts: dict[int, str] = {}
    for txt in sorted(sheet_dir.glob("sheet_*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            num, _, text = line.partition("\t")
            text = text.strip()
            texts[int(num)] = "" if text == ILLEGIBLE else text
    return texts


def ingest(sheet_dir: Path, out: Path, labeled_by: str) -> None:
    index = {r["num"]: r for r in map(json.loads, (sheet_dir / "index.jsonl").open(encoding="utf-8"))}
    texts = read_transcriptions(sheet_dir)
    missing = sorted(set(index) - set(texts))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for num in sorted(texts):
            r = index[num]
            f.write(json.dumps({"image_path": r["image_path"], "text": texts[num],
                                "source_document": r["source_document"], "line_index": r["line_index"],
                                "source_image": r["source_image"], "labeled_by": labeled_by},
                               ensure_ascii=False) + "\n")
    print(f"{len(texts)} labels ({sum(1 for t in texts.values() if not t)} illegible) → {out}; "
          f"{len(missing)} crops not yet transcribed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("manifest", type=Path)
    m.add_argument("out_dir", type=Path)
    m.add_argument("--allow-sealed", action="store_true")
    m.add_argument("--limit", type=int, default=0)
    i = sub.add_parser("ingest")
    i.add_argument("sheet_dir", type=Path)
    i.add_argument("out", type=Path)
    i.add_argument("--labeled-by", default="claude-opus-5-5")
    args = p.parse_args()
    if args.cmd == "make":
        make(args.manifest, args.out_dir, args.allow_sealed, args.limit)
    else:
        ingest(args.sheet_dir, args.out, args.labeled_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
