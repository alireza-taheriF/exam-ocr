"""Download 50 fixed-seed DastKhat samples (public CC-BY handwriting) into the harness' dir format.

    .venv/bin/python bakeoff/fetch_dastkhat.py
Writes bakeoff/dastkhat_50/{images,truth}/ and bakeoff/manual_20/01.png..20.png (first 20 of the 50).
"""
import csv
import json
import random
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

BASE = "https://huggingface.co/datasets/MrFarahmand/DastKhat/resolve/main/"
OUT = Path(__file__).parent


def main() -> None:
    rows = list(csv.DictReader(requests.get(BASE + "data/metadata.csv", timeout=60).content.decode("utf-8-sig").splitlines()))
    picked = random.Random(0).sample(rows, 50)
    for d in ("dastkhat_50/images", "dastkhat_50/truth", "manual_20"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    manual = {}
    for i, r in enumerate(picked, 1):
        stem = f"dk_{int(r['sample_id']):06d}"
        resp = requests.get(BASE + "data/" + r["file_name"], timeout=60)
        resp.raise_for_status()
        raw = resp.content
        (OUT / "dastkhat_50/images" / f"{stem}.jpg").write_bytes(raw)
        (OUT / "dastkhat_50/truth" / f"{stem}.txt").write_text(r["text"].strip(), encoding="utf-8")
        if i <= 20:
            Image.open(BytesIO(raw)).convert("RGB").save(OUT / "manual_20" / f"{i:02d}.png")
            manual[f"{i:02d}"] = f"bakeoff/dastkhat_50/images/{stem}.jpg"
    (OUT / "manual_20" / "map.json").write_text(json.dumps(manual, indent=1))
    print(f"{len(picked)} samples, {len({r['participant_id'] for r in picked})} writers")


if __name__ == "__main__":
    main()
