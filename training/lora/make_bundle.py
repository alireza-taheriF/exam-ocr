#!/usr/bin/env python3
"""Zip the local crops a Colab LoRA run needs, for upload to Google Drive.

Each manifest (image_path + text JSONL) is copied into the zip as
``manifests/<name>.jsonl`` with ``image_path`` rewritten to ``crops/<file>``,
and its crops go to ``crops/``. Unlabeled rows are dropped. Any sealed crop
(gold, test split, eval/new_gold) aborts the bundle — nothing sealed may reach
training (training/sealed.py).

    .venv-qwen/bin/python training/lora/make_bundle.py \
        --manifest exports/corrections_v1/train.jsonl dataset/val_calib.jsonl \
        --out outputs/colab/exam_ocr_lora_data.zip

On Colab: unzip to /content/data and pass /content/data/manifests/*.jsonl to
train_qwen_lora.py --local.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.lora.data import load_local  # noqa: E402
from training.sealed import sealed_crop_names  # noqa: E402


def build(manifests: list[Path], out: Path) -> dict:
    sealed = sealed_crop_names()
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        seen: set[str] = set()
        for m in manifests:
            rows = load_local(m, sealed)  # raises SealedCropError on any sealed crop
            lines = []
            for r in rows:
                src = Path(r["image_path"])
                name = f"crops/{src.name}"
                if name not in seen:
                    z.write(src, name)
                    seen.add(name)
                lines.append(json.dumps({"image_path": name, "text": r["text"]}, ensure_ascii=False))
            stem = m.stem if m.stem != "train" else f"{m.parent.name}_train"
            z.writestr(f"manifests/{stem}.jsonl", "\n".join(lines) + "\n")
            summary[stem] = len(rows)
        z.writestr("bundle_summary.json", json.dumps(
            {"manifests": summary, "crops": len(seen), "sealed_checked": len(sealed)}, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "colab" / "exam_ocr_lora_data.zip")
    args = p.parse_args()
    try:
        summary = build(args.manifest, args.out)
    except Exception:
        args.out.unlink(missing_ok=True)  # never leave a half-written bundle to upload
        raise
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB): {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
