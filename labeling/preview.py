"""Contact sheet of random crops per candidate document: python -m labeling.preview -> labeling/preview.png

Candidates = the gold documents + every document with >= 200 lines. Prints the Persian-letter ratio of
the older pseudo-labels (train/val/test.jsonl, local only) as a hint for handwritten-vs-printed / Persian-vs-English.
"""
import json
import random
import re
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont

from labeling.privacy import ROOT, _rows

FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
PERSIAN, LATIN = re.compile(r"[؀-ۿ]"), re.compile(r"[A-Za-z]")


def main() -> None:
    lines = _rows("lines_v3.jsonl")
    count = Counter(r["source_document"] for r in lines)
    gold = {r["source_document"] for r in _rows("gold.jsonl")}
    docs = sorted((d for d in count if d in gold or count[d] >= 200), key=lambda d: -count[d])
    by = defaultdict(list)
    for r in lines:
        by[r["source_document"]].append(r)
    doc_of = {r["source_image"]: r["source_document"] for r in lines}
    ratio = defaultdict(lambda: [0, 0])
    for f in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for r in _rows(f):
            t, k = r.get("text") or "", doc_of.get(r.get("source_image"), "")
            ratio[k][0] += len(PERSIAN.findall(t))
            ratio[k][1] += len(LATIN.findall(t))
    rng, font, W, H = random.Random(0), ImageFont.truetype(FONT, 22), 1500, 64
    sheet = Image.new("RGB", (W, len(docs) * (6 * H + 44)), "white")
    d, y = ImageDraw.Draw(sheet), 0
    for i, doc in enumerate(docs, 1):
        p, l = ratio[doc]
        hint = f"old labels: persian {p} / latin {l}" if p + l else "no old labels"
        d.rectangle((0, y, W, y + 38), fill="#223")
        d.text((10, y + 6), f"{i}.  {count[doc]} lines{'  [GOLD]' if doc in gold else ''}  |  {hint}", fill="white", font=font, direction="ltr")
        d.text((W - 10, y + 6), doc[:60], fill="#9cf", font=font, direction="rtl", anchor="ra")
        y += 44
        for r in rng.sample(by[doc], 5):
            im = Image.open(ROOT / r["image_path"]).convert("RGB")
            im.thumbnail((W - 20, H - 4))
            sheet.paste(im, (10, y))
            d.line((0, y + H - 2, W, y + H - 2), fill="#ccc")
            y += H
        y += 6 * H - 5 * H
    sheet.crop((0, 0, W, y)).save(ROOT / "labeling" / "preview.png")
    print(f"{len(docs)} documents -> labeling/preview.png")
    for i, doc in enumerate(docs, 1):
        p, l = ratio[doc]
        print(f"{i}. {count[doc]:5d} lines  persian/latin letters in old labels: {p}/{l}  {doc[:50]}")


if __name__ == "__main__":
    main()
