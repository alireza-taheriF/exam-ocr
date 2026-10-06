#!/usr/bin/env python3
"""Deterministic split of the hand-labeled gold set.

Default: split **by ``source_image`` (page)** so no page appears in more than
one split. Optional ``--split-by document`` keeps all pages of a
``source_document`` in the same split (avoids writer/document leakage).

Ratios default to 70% / 15% / 15% → ``train`` / ``val`` / ``gold_test``.

``gold_test`` must never be used for training, tuning, threshold selection, or
debugging decisions — only for final reporting.

Usage:
    python -m eval.split
    python -m eval.split --split-by document --out-dir eval/splits_by_document
    python -m eval.split --seed 42 --ratios 0.70 0.15 0.15
    python -m eval.split --audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLD = PROJECT_ROOT / "dataset" / "gold.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "eval" / "splits"

SPLITS = ("train", "val", "gold_test")
SplitBy = Literal["page", "document"]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_key(row: dict) -> str:
    return row.get("source_image") or row.get("image_path") or ""


def document_key(row: dict) -> str:
    return (row.get("source_document") or "").strip() or page_key(row)


def stable_page_hash(page: str) -> str:
    return hashlib.sha1(page.encode("utf-8")).hexdigest()


def assign_units(
    units: list[str],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each unit id to train/val/gold_test. Deterministic for a given seed."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    units_sorted = sorted(units, key=stable_page_hash)
    rng = random.Random(seed)
    rng.shuffle(units_sorted)

    n = len(units_sorted)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_val += n_test
        n_test = 0
    # With few units (e.g. 4 documents), rounding can wipe gold_test — keep a
    # non-empty hold-out whenever n >= 3.
    if n >= 3 and n_test == 0:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
            n_test = 1
        elif n_val > 1:
            n_val -= 1
            n_test = 1
        elif n_train > 1:
            n_train -= 1
            n_test = 1
    if n >= 3 and n_val == 0 and n_train > 1:
        n_train -= 1
        n_val = 1

    assignment: dict[str, str] = {}
    for i, unit in enumerate(units_sorted):
        if i < n_train:
            assignment[unit] = "train"
        elif i < n_train + n_val:
            assignment[unit] = "val"
        else:
            assignment[unit] = "gold_test"
    return assignment


def assign_pages(
    pages: list[str],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each page to train/val/gold_test. Deterministic for a given seed."""
    return assign_units(pages, ratios=ratios, seed=seed)


def build_split_manifests(
    rows: list[dict],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    labeled_only: bool = False,
    split_by: SplitBy = "page",
) -> dict[str, dict]:
    """Return {split_name: manifest_dict} with page-grouped rows."""
    if labeled_only:
        rows = [r for r in rows if (r.get("text") or "").strip()]

    by_page: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = page_key(row)
        if not key:
            continue
        by_page[key].append(row)

    for page in by_page:
        by_page[page].sort(key=lambda r: (r.get("line_index", 0), r.get("image_path", "")))

    if split_by == "page":
        assignment = assign_pages(list(by_page.keys()), ratios=ratios, seed=seed)
        group_by = "source_image"
    elif split_by == "document":
        # Map each page → its document, assign documents, then expand to pages.
        page_to_doc = {
            page: document_key(page_rows[0]) for page, page_rows in by_page.items()
        }
        docs = sorted(set(page_to_doc.values()))
        doc_assignment = assign_units(docs, ratios=ratios, seed=seed)
        assignment = {page: doc_assignment[doc] for page, doc in page_to_doc.items()}
        group_by = "source_document"
    else:
        raise ValueError(f"unknown split_by: {split_by}")

    manifests: dict[str, dict] = {}
    for split in SPLITS:
        pages = sorted(p for p, s in assignment.items() if s == split)
        items = []
        for page in pages:
            page_rows = by_page[page]
            items.append(
                {
                    "page": page,
                    "source_document": page_rows[0].get("source_document"),
                    "n_lines": len(page_rows),
                    "n_labeled_lines": sum(
                        1 for r in page_rows if (r.get("text") or "").strip()
                    ),
                    "lines": page_rows,
                }
            )
        manifests[split] = {
            "split": split,
            "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "gold_test": ratios[2]},
            "group_by": group_by,
            "split_by": split_by,
            "n_pages": len(pages),
            "n_lines": sum(it["n_lines"] for it in items),
            "n_labeled_lines": sum(it["n_labeled_lines"] for it in items),
            "pages": items,
            "policy": (
                "gold_test is hold-out only: never use for training, tuning, "
                "threshold selection, or debugging decisions."
                if split == "gold_test"
                else "May be used for training (train) or tuning (val)."
            ),
        }
    return manifests


def save_manifests(manifests: dict[str, dict], out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, manifest in manifests.items():
        path = out_dir / f"{split}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        paths[split] = path

    first = next(iter(manifests.values()))
    root = PROJECT_ROOT.resolve()
    summary = {
        "seed": first["seed"],
        "ratios": first["ratios"],
        "group_by": first.get("group_by", "source_image"),
        "split_by": first.get("split_by", "page"),
        "splits": {
            name: {
                "path": str(paths[name].resolve().relative_to(root)),
                "n_pages": m["n_pages"],
                "n_lines": m["n_lines"],
                "n_labeled_lines": m["n_labeled_lines"],
            }
            for name, m in manifests.items()
        },
        "leakage_check": _leakage_check(manifests),
        "document_leakage_check": _document_leakage_check(manifests),
    }
    summary_path = out_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    paths["summary"] = summary_path
    return paths


def _leakage_check(manifests: dict[str, dict]) -> dict:
    sets = {
        name: {p["page"] for p in m["pages"]} for name, m in manifests.items()
    }
    overlaps = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sorted(sets[a] & sets[b])
            overlaps[f"{a}_vs_{b}"] = {"n_overlap": len(inter), "pages": inter}
    return {"ok": all(v["n_overlap"] == 0 for v in overlaps.values()), "overlaps": overlaps}


def _document_leakage_check(manifests: dict[str, dict]) -> dict:
    """Report whether the same source_document appears in more than one split."""
    doc_to_splits: dict[str, set[str]] = defaultdict(set)
    doc_page_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, m in manifests.items():
        for p in m["pages"]:
            doc = p.get("source_document") or ""
            doc_to_splits[doc].add(name)
            doc_page_counts[doc][name] += 1
    leaked = {
        doc: {
            "splits": sorted(splits),
            "pages_per_split": dict(doc_page_counts[doc]),
        }
        for doc, splits in doc_to_splits.items()
        if len(splits) > 1
    }
    return {
        "ok": len(leaked) == 0,
        "n_documents": len(doc_to_splits),
        "n_leaked_documents": len(leaked),
        "leaked_documents": leaked,
    }


def audit_gold(rows: list[dict]) -> dict:
    """Data audit for gold.jsonl: per-document counts, lengths, empties."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[document_key(row)].append(row)

    lengths = [len((r.get("text") or "").strip()) for r in rows]
    empty = sum(1 for L in lengths if L == 0)
    # No dedicated illegible flag in gold; empty text is the proxy used in scoring.
    labeled = [L for L in lengths if L > 0]
    labeled_sorted = sorted(labeled)

    def _pct(xs: list[int], q: float) -> float | None:
        if not xs:
            return None
        return float(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))])

    per_doc = {}
    for doc, doc_rows in sorted(by_doc.items(), key=lambda kv: -len(kv[1])):
        pages = {page_key(r) for r in doc_rows}
        texts = [(r.get("text") or "").strip() for r in doc_rows]
        writers = sorted(
            {
                str(r.get(k)).strip()
                for r in doc_rows
                for k in ("writer", "author", "handwriting_id")
                if r.get(k)
            }
        )
        per_doc[doc] = {
            "n_lines": len(doc_rows),
            "n_pages": len(pages),
            "n_labeled": sum(1 for t in texts if t),
            "n_empty": sum(1 for t in texts if not t),
            "writer": writers[0] if len(writers) == 1 else (writers or None),
            "mean_text_len": (
                round(sum(len(t) for t in texts if t) / max(1, sum(1 for t in texts if t)), 2)
                if any(texts)
                else 0.0
            ),
        }

    return {
        "n_rows": len(rows),
        "n_documents": len(by_doc),
        "n_pages": len({page_key(r) for r in rows}),
        "n_empty_or_illegible": empty,
        "n_labeled": len(rows) - empty,
        "writer_known": any(
            r.get(k) for r in rows for k in ("writer", "author", "handwriting_id")
        ),
        "text_length": {
            "min": min(lengths) if lengths else 0,
            "p25": _pct(sorted(lengths), 0.25),
            "p50": _pct(sorted(lengths), 0.50),
            "p75": _pct(sorted(lengths), 0.75),
            "p90": _pct(sorted(lengths), 0.90),
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 3) if lengths else 0.0,
            "labeled_only": {
                "min": min(labeled) if labeled else 0,
                "p50": _pct(labeled_sorted, 0.50),
                "p90": _pct(labeled_sorted, 0.90),
                "max": max(labeled) if labeled else 0,
                "mean": round(sum(labeled) / len(labeled), 3) if labeled else 0.0,
            },
        },
        "lines_per_source_document": per_doc,
        "note": (
            "gold.jsonl has no explicit writer/illegible fields; empty text "
            "(72 rows historically) is treated as illegible/unscored."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "GOLD_TEST"),
    )
    p.add_argument(
        "--split-by",
        choices=["page", "document"],
        default="page",
        help="page (default): no page in two splits. document: keep whole docs together.",
    )
    p.add_argument(
        "--labeled-only",
        action="store_true",
        help="Drop empty-text gold rows before splitting (pages still stay intact).",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Print data audit (lines/doc, empties, length dist) and exit.",
    )
    args = p.parse_args(argv)

    rows = load_jsonl(args.gold)

    if args.audit:
        audit = audit_gold(rows)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        # Also check current default splits if present.
        default_summary = DEFAULT_OUT_DIR / "split_summary.json"
        if default_summary.exists():
            summary = json.loads(default_summary.read_text(encoding="utf-8"))
            print("\n--- existing page-split document leakage ---")
            print(json.dumps(summary.get("document_leakage_check") or _document_leakage_check(
                {s: json.loads((DEFAULT_OUT_DIR / f"{s}.json").read_text(encoding="utf-8"))
                 for s in SPLITS if (DEFAULT_OUT_DIR / f"{s}.json").exists()}
            ), ensure_ascii=False, indent=2))
        return 0

    # Document split with only 4 docs cannot hit 70/15/15 page ratios; still write it.
    manifests = build_split_manifests(
        rows,
        ratios=(args.ratios[0], args.ratios[1], args.ratios[2]),
        seed=args.seed,
        labeled_only=args.labeled_only,
        split_by=args.split_by,
    )
    paths = save_manifests(manifests, args.out_dir)

    print(f"Deterministic {args.split_by}-level split written:")
    for split in SPLITS:
        m = manifests[split]
        docs = sorted({p.get("source_document") for p in m["pages"]})
        print(
            f"  {split:10s}  pages={m['n_pages']:3d}  "
            f"lines={m['n_lines']:4d}  labeled={m['n_labeled_lines']:4d}  "
            f"docs={len(docs)}  → {paths[split]}"
        )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    print(f"  page_leakage_ok={summary['leakage_check']['ok']}")
    print(
        f"  document_leakage_ok={summary['document_leakage_check']['ok']}  "
        f"leaked_docs={summary['document_leakage_check']['n_leaked_documents']}"
    )
    print(f"  summary={paths['summary']}")
    print(
        "NOTE: gold_test must never be used for training, tuning, "
        "threshold selection, or debugging decisions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
