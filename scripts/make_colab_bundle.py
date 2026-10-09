#!/usr/bin/env python3
"""Pack training manifests + only the crop images they reference into one zip for Colab.

Code travels via ``git clone``; this zip carries the gitignored data. Sealed
``eval/new_gold`` crops are refused so they can never reach a training machine.

Usage:
  python scripts/make_colab_bundle.py \\
      --manifests dataset/qwen_ft/train_mix_indomain.jsonl dataset/qwen_ft/val_gold.jsonl \\
      --out outputs/colab/exam_ocr_data.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEALED = ROOT / "eval" / "new_gold" / "manifest.jsonl"


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "colab" / "exam_ocr_data.zip")
    args = p.parse_args(argv)

    sealed = {r["image_path"] for r in _rows(SEALED)} if SEALED.exists() else set()
    images: set[str] = set()
    for m in args.manifests:
        m = m if m.is_absolute() else ROOT / m
        for r in _rows(m):
            images.add(r["image_path"])
    leaked = images & sealed
    if leaked:
        print(f"REFUSING: {len(leaked)} sealed new_gold crops are in the manifests", file=sys.stderr)
        return 2
    missing = [i for i in images if not (ROOT / i).exists()]
    if missing:
        print(f"REFUSING: {len(missing)} images missing, e.g. {missing[0]}", file=sys.stderr)
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for m in args.manifests:
            m = m if m.is_absolute() else ROOT / m
            z.write(m, m.relative_to(ROOT).as_posix())
        for i in sorted(images):
            z.write(ROOT / i, i)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(images)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
