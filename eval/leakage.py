#!/usr/bin/env python3
"""Leakage audit between a training manifest and eval splits.

Checks exact image SHA-256, perceptual hash near-duplicates, exact / near-
duplicate normalized text (similarity >= 0.9), and overlap by source_document /
source page / notebook. Also summarizes training-manifest composition.

Usage:
  python -m eval.leakage \\
      --train dataset/qwen_ft/train_mix_indomain.jsonl \\
      --splits val gold_test \\
      --split-dir eval/splits_by_document
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import sentence_similarity  # noqa: E402
from eval.normalize import normalize_text  # noqa: E402

DEFAULT_TRAIN = PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix_indomain.jsonl"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "eval" / "splits_by_document"
HUMAN_SOURCES = frozenset({"gold", "gold_oversample"})
TEXT_NEAR_THRESH = 0.9
PHASH_HAMMING_MAX = 6  # near-duplicate threshold for 64-bit aHash


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> int:
    """Simple average hash (aHash), 64-bit by default. No ImageHash dependency."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    mean = sum(pixels) / max(len(pixels), 1)
    bits = 0
    for i, px in enumerate(pixels):
        if px >= mean:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def notebook_of(row: dict) -> str:
    return (row.get("source_document") or row.get("region") or "unknown").strip() or "unknown"


def page_of(row: dict) -> str:
    return (row.get("source_image") or row.get("page") or "").strip()


def load_split_lines(split_dir: Path, split: str) -> list[dict]:
    meta = json.loads((split_dir / f"{split}.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for page in meta.get("pages") or []:
        for ln in page.get("lines") or []:
            row = dict(ln)
            row.setdefault("source_document", page.get("source_document"))
            row.setdefault("source_image", page.get("page") or ln.get("source_image"))
            row["split"] = split
            out.append(row)
    return out


@dataclass
class IndexedManifest:
    path: Path
    rows: list[dict]
    by_image: dict[str, list[dict]] = field(default_factory=dict)
    by_sha: dict[str, list[str]] = field(default_factory=dict)
    by_phash: dict[int, list[str]] = field(default_factory=dict)
    by_text: dict[str, list[str]] = field(default_factory=dict)
    docs: set[str] = field(default_factory=set)
    pages: set[str] = field(default_factory=set)
    notebooks: set[str] = field(default_factory=set)


def index_manifest(path: Path, project_root: Path, *, compute_hashes: bool = True) -> IndexedManifest:
    rows = load_jsonl(path)
    idx = IndexedManifest(path=path, rows=rows)
    for row in rows:
        ip = row.get("image_path") or ""
        idx.by_image.setdefault(ip, []).append(row)
        idx.docs.add(notebook_of(row))
        pg = page_of(row)
        if pg:
            idx.pages.add(pg)
        idx.notebooks.add(notebook_of(row))
        text_n = normalize_text(row.get("text") or "")
        if text_n:
            idx.by_text.setdefault(text_n, []).append(ip)
        if not compute_hashes or not ip:
            continue
        full = project_root / ip
        if not full.exists():
            continue
        try:
            digest = sha256_file(full)
            idx.by_sha.setdefault(digest, []).append(ip)
            ph = average_hash(full)
            idx.by_phash.setdefault(ph, []).append(ip)
        except OSError:
            continue
    return idx


def composition_report(rows: list[dict]) -> dict:
    by_source = Counter(r.get("source") or "unknown" for r in rows)
    human = sum(by_source[s] for s in HUMAN_SOURCES if s in by_source)
    pseudo = sum(v for k, v in by_source.items() if "pseudo" in k)
    other = len(rows) - human - pseudo
    by_doc_source: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_doc_source[notebook_of(r)][r.get("source") or "unknown"] += 1
    return {
        "n_rows": len(rows),
        "n_unique_images": len({r.get("image_path") for r in rows if r.get("image_path")}),
        "by_source": dict(by_source),
        "human_vs_pseudo": {
            "human_gold_or_oversample": human,
            "pseudo": pseudo,
            "other": other,
        },
        "per_document": {
            doc: dict(cnt) for doc, cnt in sorted(by_doc_source.items(), key=lambda x: x[0])
        },
    }


@dataclass
class OverlapHit:
    category: str
    eval_image: str
    train_image: str
    detail: str
    eval_doc: str = ""
    train_doc: str = ""


def audit_against(
    train: IndexedManifest,
    eval_rows: list[dict],
    project_root: Path,
    *,
    text_thresh: float = TEXT_NEAR_THRESH,
    phash_max: int = PHASH_HAMMING_MAX,
) -> dict:
    hits: list[OverlapHit] = []
    eval_docs = {notebook_of(r) for r in eval_rows}
    train_docs = set(train.docs)
    doc_overlap = sorted(eval_docs & train_docs)

    # Document / page / notebook overlap (structural)
    for r in eval_rows:
        doc = notebook_of(r)
        if doc in train.docs:
            hits.append(
                OverlapHit(
                    "source_document",
                    r.get("image_path") or "",
                    "",
                    f"document={doc}",
                    eval_doc=doc,
                    train_doc=doc,
                )
            )
        pg = page_of(r)
        if pg and pg in train.pages:
            hits.append(
                OverlapHit(
                    "source_page",
                    r.get("image_path") or "",
                    "",
                    f"page={pg}",
                    eval_doc=doc,
                )
            )

    # Exact image path
    for r in eval_rows:
        ip = r.get("image_path") or ""
        if ip in train.by_image:
            hits.append(
                OverlapHit(
                    "exact_image_path",
                    ip,
                    ip,
                    "same image_path in train manifest",
                    eval_doc=notebook_of(r),
                    train_doc=notebook_of(train.by_image[ip][0]),
                )
            )

    # Exact SHA / phash / text
    for r in eval_rows:
        ip = r.get("image_path") or ""
        full = project_root / ip
        if not ip or not full.exists():
            continue
        try:
            digest = sha256_file(full)
            if digest in train.by_sha:
                for tip in train.by_sha[digest]:
                    if tip == ip:
                        continue  # already counted as path
                    hits.append(
                        OverlapHit(
                            "exact_image_hash",
                            ip,
                            tip,
                            f"sha256={digest[:16]}…",
                            eval_doc=notebook_of(r),
                        )
                    )
            ph = average_hash(full)
            # exact same aHash bucket
            for tip in train.by_phash.get(ph, []):
                if tip == ip:
                    continue
                hits.append(
                    OverlapHit(
                        "phash_near_duplicate",
                        ip,
                        tip,
                        f"hamming=0 ahash={ph:016x}",
                        eval_doc=notebook_of(r),
                    )
                )
            # near neighbors (scan train hashes — O(n) per eval; fine for <10k)
            for tph, tips in train.by_phash.items():
                dist = hamming(ph, tph)
                if 0 < dist <= phash_max:
                    for tip in tips:
                        hits.append(
                            OverlapHit(
                                "phash_near_duplicate",
                                ip,
                                tip,
                                f"hamming={dist} ahash_eval={ph:016x} ahash_train={tph:016x}",
                                eval_doc=notebook_of(r),
                            )
                        )
        except OSError:
            pass

        text_n = normalize_text(r.get("text") or "")
        if not text_n:
            continue
        if text_n in train.by_text:
            for tip in train.by_text[text_n]:
                hits.append(
                    OverlapHit(
                        "exact_text",
                        ip,
                        tip,
                        f"text={text_n[:80]}",
                        eval_doc=notebook_of(r),
                    )
                )
        # near-duplicate text vs unique train texts (skip tiny tokens)
        if len(text_n) >= 4:
            for ttext, tips in train.by_text.items():
                if ttext == text_n or len(ttext) < 4:
                    continue
                sim = sentence_similarity(ttext, text_n)
                if sim >= text_thresh:
                    for tip in tips[:1]:
                        hits.append(
                            OverlapHit(
                                "near_text",
                                ip,
                                tip,
                                f"sim={sim:.3f} eval={text_n[:60]!r} train={ttext[:60]!r}",
                                eval_doc=notebook_of(r),
                            )
                        )

    # Deduplicate hits by (category, eval_image, train_image, detail)
    seen: set[tuple] = set()
    uniq: list[OverlapHit] = []
    for h in hits:
        key = (h.category, h.eval_image, h.train_image, h.detail)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    counts = Counter(h.category for h in uniq)
    examples: dict[str, list[dict]] = defaultdict(list)
    for h in uniq:
        if len(examples[h.category]) >= 5:
            continue
        examples[h.category].append(
            {
                "eval_image": h.eval_image,
                "train_image": h.train_image,
                "detail": h.detail,
                "eval_doc": h.eval_doc,
                "train_doc": h.train_doc,
            }
        )

    return {
        "n_eval_rows": len(eval_rows),
        "n_eval_labeled": sum(1 for r in eval_rows if (r.get("text") or "").strip()),
        "eval_documents": sorted(eval_docs),
        "train_documents": sorted(train_docs),
        "documents_in_both_train_and_eval": doc_overlap,
        "overlap_counts": dict(counts),
        "examples": dict(examples),
        "n_hits_total": len(uniq),
    }


def run_audit(
    train_path: Path,
    split_dir: Path,
    splits: list[str],
    *,
    project_root: Path = PROJECT_ROOT,
    meta_path: Path | None = None,
) -> dict:
    train = index_manifest(train_path, project_root)
    comp = composition_report(train.rows)
    meta = {}
    if meta_path and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    per_split = {}
    for split in splits:
        rows = load_split_lines(split_dir, split)
        # Prefer labeled-only for text/image audit of scored set; still report all.
        labeled = [r for r in rows if (r.get("text") or "").strip()]
        per_split[split] = audit_against(train, labeled, project_root)

    return {
        "train_manifest": str(train_path),
        "composition": comp,
        "training_meta": meta,
        "splits": per_split,
    }


def print_audit(report: dict) -> None:
    print("=" * 72)
    print("LEAKAGE AUDIT")
    print("=" * 72)
    print(f"train_manifest: {report['train_manifest']}")
    comp = report["composition"]
    print(f"\nComposition: n_rows={comp['n_rows']} unique_images={comp['n_unique_images']}")
    print(f"  by_source: {comp['by_source']}")
    print(f"  human_vs_pseudo: {comp['human_vs_pseudo']}")
    print("  per_document:")
    for doc, cnt in comp["per_document"].items():
        print(f"    {doc}: {cnt}")
    meta = report.get("training_meta") or {}
    if meta:
        print("\nCheckpoint / train settings (training_meta.json):")
        for k in (
            "base_model",
            "epochs",
            "completed_steps",
            "target_steps",
            "lora_r",
            "lora_alpha",
            "learning_rate",
            "device",
            "note",
        ):
            if k in meta:
                print(f"  {k}: {meta[k]}")

    for split, audit in (report.get("splits") or {}).items():
        print(f"\n--- vs {split} (labeled n={audit['n_eval_labeled']}) ---")
        print(f"  eval docs : {audit['eval_documents']}")
        print(f"  train docs: {audit['train_documents']}")
        both = audit["documents_in_both_train_and_eval"]
        print(f"  docs in BOTH train and {split}: {both if both else '(none)'}")
        print(f"  overlap_counts: {audit['overlap_counts'] or '(none)'}")
        for cat, exs in (audit.get("examples") or {}).items():
            print(f"  examples[{cat}] ({len(exs)} shown):")
            for ex in exs:
                print(
                    f"    eval={ex['eval_image']} train={ex['train_image']} "
                    f"doc={ex.get('eval_doc')} | {ex['detail']}"
                )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--splits", nargs="+", default=["val", "gold_test"])
    p.add_argument(
        "--meta",
        type=Path,
        default=PROJECT_ROOT / "models" / "qwen3-vl-fa-exam-lora-v1" / "training_meta.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval" / "phase_c_verify" / "leakage_audit.json",
    )
    args = p.parse_args(argv)

    report = run_audit(args.train, args.split_dir, args.splits, meta_path=args.meta)
    print_audit(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
