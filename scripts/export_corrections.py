#!/usr/bin/env python3
"""Export student corrections as a line-level training/eval set for Colab.

Only lines from confirmed documents whose students consented are exported.
Trusted corrections (``learning/label_trust.py``) go to ``train``; hold-out
documents go to ``holdout`` (evaluation only, never training — see
``.claude/skills/ocr-eval-gate``). Quarantined and rejected edits are skipped.
Lines accepted without edits are included only with ``--include-unedited``
(weak labels: the student may simply not have checked them).

    python scripts/export_corrections.py --out exports/corrections_v1

Output: ``<out>/{train,holdout}.jsonl`` with ``image_path`` relative to
``<out>``, the crops under ``<out>/crops/``, and ``<out>/summary.json``.
Rows match the format of ``dataset/train_v3.jsonl`` (``image_path``, ``text``,
``source_document``) so the existing training scripts can read them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.store import Store  # noqa: E402
from learning.label_trust import TRUSTED, trust_for  # noqa: E402


def export(store: Store, out: Path, *, include_unedited: bool = False) -> dict:
    """Write ``<out>/{train,holdout}.jsonl`` + crops; return the summary counts."""
    rows = store.final_pairs(confirmed_only=True)
    if include_unedited:
        rows += store.confirmed_untouched_pairs()

    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    sinks = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in ("train", "holdout")}
    counts: Counter[str] = Counter()
    try:
        for row in rows:
            if not row["consent"]:
                counts["skipped_no_consent"] += 1
                continue
            trust = trust_for(row)
            if trust.status != TRUSTED:
                counts[f"skipped_{trust.status}"] += 1
                continue
            src = Path(row["crop_path"])
            if not src.exists():
                counts["skipped_missing_crop"] += 1
                continue
            dst = crops_dir / f"{row['line_id']}.png"
            shutil.copyfile(src, dst)
            split = "holdout" if row["holdout"] else "train"
            sinks[split].write(json.dumps({
                "image_path": str(dst.relative_to(out)),
                "text": row["final_text"],
                "source_document": row["document_id"],
                "line_id": row["line_id"],
                "predicted_text": row["predicted_text"],
                "raw_text": row["raw_text"],
                "model_version": row["model_version"],
                "edited": bool(row["version"]),
                "admin_review": row.get("review"),
                "label_backend": "student_correction",
            }, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for f in sinks.values():
            f.close()
    summary = dict(counts)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path,
                    default=Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")) / "exam_ocr.sqlite3")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-unedited", action="store_true")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 1
    summary = {"db": str(args.db), **export(Store(args.db), args.out, include_unedited=args.include_unedited)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
