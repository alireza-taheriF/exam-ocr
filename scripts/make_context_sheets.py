#!/usr/bin/env python3
"""Contact sheets with page context: each item = enlarged crop + surrounding page region.

For every crop the source page is cut a few line-heights above/below and a bit to the
sides, the crop's own box is outlined in red, and the crop itself is shown enlarged on top.
Labelers transcribe ONLY the crop (see docs/transcription_rules.md).

    make_context_sheets.py --todo X.jsonl --done Y.jsonl --outdir DIR --per 3
Sidecar sheet_NNN.json maps index -> image_path (same format as make_label_sheets.py).
"""
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W = 1100
MARGIN = 60


def load(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()] if Path(p).exists() else []


def enlarge(img, max_w=W, target_h=110, max_h=240):
    w, h = img.size
    s = min(max_w / w, max_h / h)
    s = max(s, min(target_h / h, max_w / w, 3.0)) if h * s < target_h else s
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def item(meta):
    crop = Image.open(ROOT / meta["image_path"]).convert("RGB")
    page = Image.open(ROOT / meta["source_image"]).convert("RGB")
    x0, y0, x1, y1 = meta["crop_bbox"]
    h = max(y1 - y0, 24)
    cx0, cx1 = max(0, x0 - 120), min(page.width, x1 + 120)
    cy0, cy1 = max(0, y0 - int(2.5 * h)), min(page.height, y1 + int(2.5 * h))
    ctx = page.crop((cx0, cy0, cx1, cy1))
    d = ImageDraw.Draw(ctx)
    d.rectangle((x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0), outline=(255, 0, 0), width=max(2, h // 12))
    s = min(W / ctx.width, 1.6, 380 / ctx.height)
    ctx = ctx.resize((max(1, int(ctx.width * s)), max(1, int(ctx.height * s))), Image.LANCZOS)
    top = enlarge(crop)
    out = Image.new("RGB", (W, top.height + ctx.height + 14), "white")
    out.paste(top, (0, 0))
    out.paste(ctx, (0, top.height + 8))
    ImageDraw.Draw(out).line([(0, top.height + 4), (W, top.height + 4)], fill=(0, 120, 255), width=2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--todo", required=True)
    p.add_argument("--done")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--per", type=int, default=3)
    a = p.parse_args()
    lines = {r["image_path"]: r for r in load(ROOT / "dataset/lines_v3.jsonl")}
    done = {r["image_path"] for r in load(a.done)} if a.done else set()
    todo = [r for r in load(a.todo) if r["image_path"] not in done]
    a.outdir.mkdir(parents=True, exist_ok=True)
    for old in a.outdir.glob("sheet_*"):
        old.unlink()
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except OSError:
        font = ImageFont.load_default()
    for k in range(0, len(todo), a.per):
        chunk = todo[k:k + a.per]
        tiles = [item(lines[r["image_path"]]) for r in chunk]
        sheet = Image.new("RGB", (MARGIN + W, sum(t.height + 16 for t in tiles)), "white")
        d = ImageDraw.Draw(sheet)
        y = 0
        for i, t in enumerate(tiles):
            d.text((8, y + 10), str(i + 1), fill="red", font=font)
            sheet.paste(t, (MARGIN, y))
            y += t.height + 16
            d.line([(0, y - 8), (sheet.width, y - 8)], fill=(120, 120, 120), width=3)
        n = k // a.per
        sheet.save(a.outdir / f"sheet_{n:03d}.png")
        (a.outdir / f"sheet_{n:03d}.json").write_text(
            json.dumps({str(i + 1): r["image_path"] for i, r in enumerate(chunk)}, ensure_ascii=False))
    print(f"{len(todo)} todo -> {-(-len(todo) // a.per)} sheets")


main()
