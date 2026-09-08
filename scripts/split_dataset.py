#!/usr/bin/env python3
"""Split line-level data into train/val/test with NO document crossing splits.

Replaces the flat shuffle in `scripts/split_dataset_legacy.py`, which leaked
876/878 val pages and 865/869 test pages into train.

## Why we balance by line count rather than by document count

There are only 12 source documents and they are wildly unequal — the largest
supplies ~615 lines, the smallest ~32. A 70/15/15 split *by document count*
(8/2/2 documents) can therefore land anywhere between ~40% and ~90% of the lines
in train depending on which documents fall where, and a 2-document val set gives
a CER estimate dominated by one writer's handwriting.

So we split by document but optimise for the *line-count* target: documents are
sorted largest-first and greedily assigned to whichever split is furthest below
its quota (largest-remainder / LPT bin-packing). Documents stay intact, and the
realised line ratios land close to 70/15/15. The cost is that with 12 documents
the ratios cannot be hit exactly — the report prints the realised split so the
deviation is visible rather than assumed.

Note that even a correct document-level split cannot fix corpus narrowness: 12
documents means val/test each see 1-3 unseen writers. Treat the resulting CER as
a lower bound on real-world error, not an unbiased estimate.

Usage:
    python scripts/split_dataset.py --input dataset/lines_v3.jsonl
    python scripts/split_dataset.py --input dataset/lines_v3.jsonl --group-by source_image

Writes dataset/{train,val,test}_v3.jsonl — the old flat-shuffled train.jsonl /
val.jsonl / test.jsonl are left untouched as the reference the audit cites.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "lines_v3.jsonl"
# Written under *_v3 names rather than over dataset/train.jsonl. Two reasons: the
# old flat-shuffled splits stay on disk as the reference the audit refers to, and
# — more importantly — nothing that already reads dataset/train.jsonl can pick up
# a manifest whose labels are still empty. Point Phase 2 at these explicitly.
TRAIN_OUT = PROJECT_ROOT / "dataset" / "train_v3.jsonl"
VAL_OUT = PROJECT_ROOT / "dataset" / "val_v3.jsonl"
TEST_OUT = PROJECT_ROOT / "dataset" / "test_v3.jsonl"
LEAKAGE_REPORT = PROJECT_ROOT / "dataset" / "split_leakage_report.json"

SPLITS = ("train", "val", "test")

# A group with fewer lines than this is a fragment — mostly the single-page
# crawled web images, which contribute ~10% of val/test. Counting only the
# groups above it shows how many *substantial* documents (writers) each split
# actually sees, which is what determines whether its CER is stable.
SUBSTANTIVE_GROUP_MIN_LINES = 50


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    missing = 0
    for row in rows:
        value = row.get(key)
        if not value:
            missing += 1
            # Fall back to the page, then to the row itself, so a row is never
            # silently dropped — but count it, because a missing group key means
            # the extraction script did not populate it.
            value = row.get("source_image") or row.get("image_path") or "__ungrouped__"
        groups[str(value)].append(row)
    if missing:
        print(f"  WARNING: {missing} rows had no '{key}'; fell back to source_image")
    return dict(groups)


def assign_groups(
    groups: dict[str, list[dict]],
    *,
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups to splits, balancing total line count.

    Largest-processing-time bin packing: walk groups largest-first and give each
    to the split with the largest shortfall against its line quota. Ties broken
    by a seeded shuffle so the result is deterministic but not order-dependent.
    """
    total_lines = sum(len(v) for v in groups.values())
    quota = {s: ratios[s] * total_lines for s in SPLITS}
    assigned_lines = dict.fromkeys(SPLITS, 0)
    assignment: dict[str, str] = {}

    rng = random.Random(seed)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # Shuffle within equal-size bands so seed actually changes the outcome.
    bands: dict[int, list[str]] = defaultdict(list)
    for name, members in ordered:
        bands[len(members)].append(name)
    for band in bands.values():
        rng.shuffle(band)
    ordered_names = [n for size in sorted(bands, reverse=True) for n in bands[size]]

    for name in ordered_names:
        size = len(groups[name])
        target = max(SPLITS, key=lambda s: (quota[s] - assigned_lines[s], s))
        assignment[name] = target
        assigned_lines[target] += size
    return assignment


def build_leakage_report(
    splits: dict[str, list[dict]], *, group_key: str
) -> dict:
    """Verify no group crosses a split, and quantify duplicate label strings.

    Group overlap MUST be 0 — a non-zero value means the split is invalid.
    Duplicate label text across splits is reported but is not leakage on its own:
    two different documents can legitimately contain the same short phrase.
    """
    groups = {
        s: {str(r.get(group_key) or r.get("source_image")) for r in rows}
        for s, rows in splits.items()
    }
    pages = {s: {str(r.get("source_image")) for r in rows} for s, rows in splits.items()}
    # Empty labels are excluded: a crop manifest produced with `--backend none`
    # has every `text` empty, and counting "" as a shared string would report a
    # phantom duplicate for every split pair.
    texts = {
        s: {t for t in (str(r.get("text") or "").strip() for r in rows) if t}
        for s, rows in splits.items()
    }
    paths = {s: {str(r.get("image_path")) for r in rows} for s, rows in splits.items()}
    unlabeled = sum(
        1 for rows in splits.values() for r in rows if not str(r.get("text") or "").strip()
    )

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    group_sizes = {
        s: sorted(
            (len([r for r in rows if str(r.get(group_key) or r.get("source_image")) == g])
             for g in groups[s]),
            reverse=True,
        )
        for s, rows in splits.items()
    }
    report = {
        "group_key": group_key,
        "counts": {
            s: {
                "lines": len(splits[s]),
                "groups": len(groups[s]),
                "source_pages": len(pages[s]),
                "substantive_groups": sum(
                    1 for n in group_sizes[s] if n >= SUBSTANTIVE_GROUP_MIN_LINES
                ),
                "largest_group_lines": group_sizes[s][0] if group_sizes[s] else 0,
            }
            for s in SPLITS
        },
        "ratios": {
            s: round(len(splits[s]) / max(1, sum(len(v) for v in splits.values())), 4)
            for s in SPLITS
        },
        "overlaps": {},
        "unlabeled_rows": unlabeled,
        "leakage_free": True,
    }
    for a, b in pairs:
        shared_groups = sorted(groups[a] & groups[b])
        shared_pages = sorted(pages[a] & pages[b])
        shared_crops = sorted(paths[a] & paths[b])
        shared_texts = texts[a] & texts[b]
        report["overlaps"][f"{a}__{b}"] = {
            "shared_groups": len(shared_groups),
            "shared_group_examples": shared_groups[:10],
            "shared_source_pages": len(shared_pages),
            "shared_image_paths": len(shared_crops),
            "duplicate_label_strings": len(shared_texts),
            "duplicate_label_pct_of_smaller": round(
                100 * len(shared_texts) / max(1, min(len(texts[a]), len(texts[b]))), 2
            ),
        }
        if shared_groups or shared_pages or shared_crops:
            report["leakage_free"] = False
    return report


def main() -> int:
    p = argparse.ArgumentParser(description="Document-grouped train/val/test split")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--group-by",
        default="source_document",
        choices=["source_document", "source_image"],
        help="Grouping unit. source_document is correct; source_image still lets "
        "pages of one notebook cross splits (same writer, ink, and scan settings).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        print("Run: python scripts/extract_and_align_lines.py first")
        return 1

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        print(f"Invalid ratios: test would be {test_ratio:.2f}")
        return 1
    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": test_ratio}

    rows = load_jsonl(args.input)
    if not rows:
        print("No rows in input")
        return 1

    groups = group_rows(rows, args.group_by)
    if len(groups) < 3:
        print(f"Only {len(groups)} group(s) — cannot make 3 disjoint splits.")
        return 1

    assignment = assign_groups(groups, ratios=ratios, seed=args.seed)
    splits: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for name, members in groups.items():
        splits[assignment[name]].extend(members)

    report = build_leakage_report(splits, group_key=args.group_by)

    print(f"\nInput: {args.input}  ({len(rows)} lines, {len(groups)} groups)")
    print(f"Grouped by: {args.group_by}   seed={args.seed}\n")
    print(f"  {'split':6s} {'lines':>7s} {'target':>8s} {'actual':>8s} {'groups':>7s} "
          f"{'>=50 lines':>11s}")
    for s in SPLITS:
        c = report["counts"][s]
        print(
            f"  {s:6s} {c['lines']:7d} {ratios[s]:8.1%} "
            f"{report['ratios'][s]:8.1%} {c['groups']:7d} {c['substantive_groups']:11d}"
        )
    train_docs = report["counts"]["train"]["substantive_groups"]
    if train_docs <= 5:
        print(f"\n  CAVEAT: train sees only {train_docs} substantial document(s). "
              f"Val/test CER measures generalisation to a handful of unseen\n"
              f"  writers, so treat it as a lower bound on real-world error — "
              f"widening the corpus matters more than tuning the split.")

    print("\n  Leakage check (shared groups must be 0):")
    for pair, ov in report["overlaps"].items():
        flag = "OK " if ov["shared_groups"] == 0 else "FAIL"
        print(
            f"    [{flag}] {pair:12s} groups={ov['shared_groups']:3d} "
            f"pages={ov['shared_source_pages']:4d} crops={ov['shared_image_paths']:4d} "
            f"dup-labels={ov['duplicate_label_strings']:4d} "
            f"({ov['duplicate_label_pct_of_smaller']:.1f}% of smaller)"
        )
    if report["unlabeled_rows"]:
        print(f"\n  NOTE: {report['unlabeled_rows']:,} of {len(rows):,} rows have an "
              f"empty `text`. This is a crop manifest, not trainable data — "
              f"label it before Phase 2.")

    print("\n  Per-group assignment:")
    for name in sorted(groups, key=lambda n: -len(groups[n])):
        print(f"    {assignment[name]:6s} {len(groups[name]):5d}  {name}")

    if args.dry_run:
        LEAKAGE_REPORT.parent.mkdir(parents=True, exist_ok=True)
        LEAKAGE_REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n--dry-run: split files NOT written; report → {LEAKAGE_REPORT}")
        return 0 if report["leakage_free"] else 2

    for path, s in ((TRAIN_OUT, "train"), (VAL_OUT, "val"), (TEST_OUT, "test")):
        write_jsonl(path, splits[s])
        print(f"\n  wrote {len(splits[s]):6d} → {path}")
    LEAKAGE_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  wrote report → {LEAKAGE_REPORT}")

    if not report["leakage_free"]:
        print("\n  LEAKAGE DETECTED — do not train on this split.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
