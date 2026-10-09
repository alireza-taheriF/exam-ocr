#!/usr/bin/env python3
"""Phase C-VERIFY: validate LoRA vs baseline without further training.

Runs leakage audit, paired bootstrap significance, per-line delta CSV,
held-out-document report, sealed new_gold preparation, and a verdict table.

Does NOT train. Does NOT tune hyperparameters on gold_test.

Usage:
  python -m eval.verify_lora
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.error_analysis import classify_pair  # noqa: E402
from eval.leakage import (  # noqa: E402
    composition_report,
    index_manifest,
    load_jsonl,
    load_split_lines,
    print_audit,
    run_audit,
    sha256_file,
)
from eval.metrics import bootstrap_mean_ci, sentence_similarity  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "eval" / "phase_c_verify"


def sample_gold_todo(rows: list[dict], *, target: int, seed: int) -> list[dict]:
    """Even draw across substantive notebooks (same recipe as label_gold_interactive)."""
    import collections

    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_doc[row.get("source_document", "unknown")].append(row)
    docs = {d: rs for d, rs in by_doc.items() if len(rs) >= 50}
    if not docs:
        docs = dict(by_doc)
    rng = random.Random(seed)
    per_doc = max(1, target // max(len(docs), 1))
    picked: list[dict] = []
    for doc in sorted(docs, key=lambda d: (-len(docs[d]), d)):
        pool = docs[doc]
        picked += rng.sample(pool, min(per_doc, len(pool)))
    rng.shuffle(picked)
    return picked
DEFAULT_TRAIN = PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix_indomain.jsonl"
DEFAULT_BASE_CACHE = (
    PROJECT_ROOT / "dataset" / "pred_qwen3-vl-2b-persian-arabic-ocr-v1.0.jsonl"
)
DEFAULT_LORA_VAL = PROJECT_ROOT / "outputs" / "eval" / "errors_val_line_crops_qwen_local_lora.csv"
DEFAULT_LORA_GT = PROJECT_ROOT / "outputs" / "eval" / "errors_gold_test_line_crops_qwen_local.csv"
# Fallbacks if *_lora.csv missing
FALLBACK_LORA_VAL = PROJECT_ROOT / "outputs" / "eval" / "errors_val_line_crops_qwen_local.csv"
TRAIN_MANIFESTS = [
    PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix_indomain.jsonl",
    PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix.jsonl",
    PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix_balanced.jsonl",
    PROJECT_ROOT / "dataset" / "qwen_ft" / "train_mix_fast.jsonl",
]


def load_pred_map_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in load_jsonl(path):
        ip = row.get("image_path") or ""
        if ip:
            out[ip] = row.get("text") or row.get("pred") or ""
    return out


def load_pred_map_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ip = r.get("image_path") or ""
            if not ip:
                continue
            rows[ip] = {
                "ref": r.get("ref") or r.get("ground_truth") or "",
                "pred": r.get("pred") or r.get("predicted") or "",
                "page": r.get("page") or "",
                "region": r.get("region") or "",
            }
    return rows


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    n_boot: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    return bootstrap_mean_ci(deltas, n_boot=n_boot, seed=seed, alpha=alpha)


def sign_test_p(deltas: list[float]) -> float:
    """Two-sided exact binomial sign test on nonzero deltas (p=0.5)."""
    pos = sum(1 for d in deltas if d > 1e-12)
    neg = sum(1 for d in deltas if d < -1e-12)
    n = pos + neg
    if n == 0:
        return 1.0
    # P(X <= k) + P(X >= n-k) with k = min(pos, neg), X~Bin(n,0.5)
    k = min(pos, neg)
    # Exact two-sided: 2 * sum_{i=0..k} C(n,i) / 2^n, capped at 1
    # Compute in log space for stability
    total = 0.0
    for i in range(k + 1):
        # C(n,i) / 2^n
        logc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
        total += math.exp(logc)
    p = min(1.0, 2.0 * total)
    return p


def build_paired_rows(
    split_rows: list[dict],
    baseline_preds: dict[str, str],
    lora_rows: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    for r in split_rows:
        ref = (r.get("text") or "").strip()
        if not ref:
            continue
        ip = r.get("image_path") or ""
        if ip not in baseline_preds or ip not in lora_rows:
            continue
        bpred = baseline_preds[ip]
        lpred = lora_rows[ip]["pred"]
        # Prefer split gold ref; fall back to CSV ref
        sb = sentence_similarity(bpred, ref)
        sl = sentence_similarity(lpred, ref)
        cats = classify_pair(ref, lpred)
        out.append(
            {
                "image_path": ip,
                "source_document": r.get("source_document") or "",
                "source_image": r.get("source_image") or r.get("page") or "",
                "ref": ref,
                "baseline_pred": bpred,
                "lora_pred": lpred,
                "baseline_sim": sb,
                "lora_sim": sl,
                "delta": sl - sb,
                "error_categories": "|".join(cats),
            }
        )
    return out


def write_delta_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path",
        "source_document",
        "source_image",
        "ref",
        "baseline_pred",
        "lora_pred",
        "baseline_sim",
        "lora_sim",
        "delta",
        "error_categories",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["baseline_sim"] = f"{r['baseline_sim']:.6f}"
            out["lora_sim"] = f"{r['lora_sim']:.6f}"
            out["delta"] = f"{r['delta']:.6f}"
            w.writerow(out)


def summarize_paired(rows: list[dict], *, n_boot: int = 10000) -> dict:
    deltas = [r["delta"] for r in rows]
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    boot = paired_bootstrap_ci(deltas, n_boot=n_boot)
    sorted_by_delta = sorted(rows, key=lambda r: r["delta"], reverse=True)
    top3 = sorted_by_delta[:3]
    total_pos = sum(d for d in deltas if d > 0)
    top3_share = (
        sum(r["delta"] for r in top3 if r["delta"] > 0) / total_pos if total_pos > 0 else 0.0
    )
    return {
        "n": len(rows),
        "mean_baseline_sim": sum(r["baseline_sim"] for r in rows) / len(rows) if rows else 0.0,
        "mean_lora_sim": sum(r["lora_sim"] for r in rows) / len(rows) if rows else 0.0,
        "mean_delta": boot["mean"],
        "delta_ci_low": boot["ci_low"],
        "delta_ci_high": boot["ci_high"],
        "n_boot": n_boot,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p": sign_test_p(deltas),
        "top10_gains": [
            {
                "image_path": r["image_path"],
                "delta": round(r["delta"], 4),
                "baseline_sim": round(r["baseline_sim"], 4),
                "lora_sim": round(r["lora_sim"], 4),
                "ref": r["ref"][:80],
            }
            for r in sorted_by_delta[:10]
        ],
        "top10_losses": [
            {
                "image_path": r["image_path"],
                "delta": round(r["delta"], 4),
                "baseline_sim": round(r["baseline_sim"], 4),
                "lora_sim": round(r["lora_sim"], 4),
                "ref": r["ref"][:80],
            }
            for r in sorted_by_delta[-10:][::-1]
        ],
        "top3_positive_gain_share": round(top3_share, 4),
        "ci_excludes_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
        "significant_positive": bool(boot["ci_low"] > 0),
    }


def heldout_report(train_path: Path, split_dir: Path, paired: dict[str, dict]) -> dict:
    train = index_manifest(train_path, PROJECT_ROOT, compute_hashes=False)
    train_docs = sorted(train.docs)
    all_docs: set[str] = set()
    for split in ("train", "val", "gold_test"):
        for r in load_split_lines(split_dir, split):
            all_docs.add(r.get("source_document") or "unknown")
    heldout_docs = sorted(all_docs - set(train_docs))
    note = ""
    if not heldout_docs:
        note = (
            "LoRA training manifest covers all labeled documents; "
            "a clean document-level held-out evaluation is impossible with current labeled data."
        )
    else:
        note = (
            f"Training manifest documents: {train_docs}. "
            f"Held-out labeled documents: {heldout_docs}."
        )
    per_doc = {}
    for split, summary in paired.items():
        # Recompute per-document from stored rows file if needed — filled by caller
        per_doc[split] = summary
    return {
        "train_documents": train_docs,
        "all_labeled_documents": sorted(all_docs),
        "heldout_documents": heldout_docs,
        "clean_document_heldout_available": bool(heldout_docs),
        "note": note,
        "paired_on_heldout_splits": per_doc,
    }


def prepare_new_gold(
    *,
    target: int = 300,
    seed: int = 7,
    out_dir: Path = PROJECT_ROOT / "eval" / "new_gold",
) -> dict:
    """Create sealed eval/new_gold from stratified candidates excluded from train manifests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Pool: all lines_v3 rows with a crop, stratified by notebook.
    lines_path = PROJECT_ROOT / "dataset" / "lines_v3.jsonl"
    if not lines_path.exists():
        raise FileNotFoundError(lines_path)
    pool = load_jsonl(lines_path)

    # Exclude every image path that appears in any training manifest.
    train_paths: set[str] = set()
    train_hashes: set[str] = set()
    for mp in TRAIN_MANIFESTS:
        if not mp.exists():
            continue
        for row in load_jsonl(mp):
            ip = row.get("image_path") or ""
            if not ip:
                continue
            train_paths.add(ip)
            full = PROJECT_ROOT / ip
            if full.exists():
                try:
                    train_hashes.add(sha256_file(full))
                except OSError:
                    pass

    # Also exclude already-scored gold labeled lines so the sealed set is fresh.
    existing_gold = {
        r["image_path"]
        for r in load_jsonl(PROJECT_ROOT / "dataset" / "gold.jsonl")
        if (r.get("text") or "").strip()
    }

    eligible = []
    for row in pool:
        ip = row.get("image_path") or ""
        if not ip or ip in train_paths or ip in existing_gold:
            continue
        if not (PROJECT_ROOT / ip).exists():
            continue
        eligible.append(row)

    # Write candidates export (labeling export)
    candidates_path = PROJECT_ROOT / "dataset" / "gold_candidates.jsonl"
    sampled = sample_gold_todo(eligible, target=target, seed=seed)
    if len(sampled) < target:
        rng = random.Random(seed)
        have = {r["image_path"] for r in sampled}
        rest = [r for r in eligible if r["image_path"] not in have]
        rng.shuffle(rest)
        sampled.extend(rest[: max(0, target - len(sampled))])

    # Drop any sampled crop whose bytes collide with a train image.
    filtered = []
    for r in sampled:
        full = PROJECT_ROOT / r["image_path"]
        try:
            digest = sha256_file(full)
        except OSError:
            continue
        if digest in train_hashes:
            continue
        filtered.append(r)
    sampled = filtered[:target]

    with candidates_path.open("w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cache = load_pred_map_jsonl(DEFAULT_BASE_CACHE)

    sealed_manifest = []
    corrections = []
    for i, r in enumerate(sampled):
        ip = r["image_path"]
        line_id = f"newgold_{i:04d}_{Path(ip).stem}"
        model_text = cache.get(ip, "")
        entry = {
            "line_id": line_id,
            "image_path": ip,
            "crop_path": ip,
            "source_document": r.get("source_document") or "",
            "source_image": r.get("source_image") or "",
            "model_text": model_text,
            "corrected_text": "",
            "sealed": True,
        }
        sealed_manifest.append(entry)
        corrections.append(
            {
                "line_id": line_id,
                "crop_path": ip,
                "model_text": model_text,
                "corrected_text": "",
            }
        )

    manifest_path = out_dir / "manifest.jsonl"
    corrections_path = out_dir / "corrections.csv"
    seal_path = out_dir / "SEALED.json"
    readme_path = out_dir / "README.md"

    with manifest_path.open("w", encoding="utf-8") as f:
        for e in sealed_manifest:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with corrections_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["line_id", "crop_path", "model_text", "corrected_text"]
        )
        w.writeheader()
        w.writerows(corrections)

    sealed_paths = {e["image_path"] for e in sealed_manifest}
    sealed_hashes = {sha256_file(PROJECT_ROOT / e["image_path"]) for e in sealed_manifest}

    overlaps = []
    for mp in TRAIN_MANIFESTS:
        if not mp.exists():
            continue
        for row in load_jsonl(mp):
            ip = row.get("image_path") or ""
            if ip in sealed_paths:
                overlaps.append({"manifest": str(mp), "kind": "path", "image_path": ip})
                continue
            full = PROJECT_ROOT / ip
            if not full.exists():
                continue
            try:
                if sha256_file(full) in sealed_hashes:
                    overlaps.append(
                        {"manifest": str(mp), "kind": "hash", "image_path": ip}
                    )
            except OSError:
                pass

    seal_meta = {
        "sealed": True,
        "n_lines": len(sealed_manifest),
        "target": target,
        "seed": seed,
        "candidates_export": str(candidates_path.relative_to(PROJECT_ROOT)),
        "stratified_by": "source_document (notebook)",
        "excluded_from_training_manifests": [str(p.relative_to(PROJECT_ROOT)) for p in TRAIN_MANIFESTS if p.exists()],
        "hash_check_overlaps": overlaps,
        "hash_check_ok": len(overlaps) == 0,
        "by_notebook": dict(Counter(e["source_document"] for e in sealed_manifest)),
        "policy": (
            "Do not use these crops for training or for hyperparameter selection. "
            "Fill corrections.csv corrected_text, then run: "
            "python -m eval.new_gold.eval_sealed --preds <preds.jsonl>"
        ),
    }
    seal_path.write_text(json.dumps(seal_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "# Sealed new gold set\n\n"
        "These line crops are **sealed**: excluded (by path + SHA-256) from every "
        "`dataset/qwen_ft/train_mix*.jsonl` manifest.\n\n"
        "1. Open each `crop_path` and fill `corrected_text` in `corrections.csv`.\n"
        "2. Evaluate any model:\n\n"
        "```bash\n"
        "python -m eval.new_gold.eval_sealed --preds path/to/preds.jsonl\n"
        "```\n\n"
        "Predictions JSONL must have `image_path` (or `crop_path`) and `text` (or `pred`).\n",
        encoding="utf-8",
    )

    return {
        "out_dir": str(out_dir),
        "n_candidates_pool_eligible": len(eligible),
        "n_sealed": len(sealed_manifest),
        "by_notebook": seal_meta["by_notebook"],
        "hash_check_ok": seal_meta["hash_check_ok"],
        "hash_check_overlaps": overlaps,
        "candidates_export": str(candidates_path),
    }


def print_paired(split: str, summary: dict) -> None:
    print(f"\n=== Paired significance: {split} ===")
    print(f"  n={summary['n']}")
    print(f"  mean baseline sim : {summary['mean_baseline_sim']:.4f}")
    print(f"  mean LoRA sim     : {summary['mean_lora_sim']:.4f}")
    print(
        f"  mean delta        : {summary['mean_delta']:+.4f} "
        f"95% CI [{summary['delta_ci_low']:+.4f}, {summary['delta_ci_high']:+.4f}] "
        f"(boot={summary['n_boot']})"
    )
    print(f"  win / tie / loss  : {summary['wins']} / {summary['ties']} / {summary['losses']}")
    print(f"  sign-test p       : {summary['sign_test_p']:.6g}")
    print(f"  top-3 share of +gain: {summary['top3_positive_gain_share']:.1%}")
    print("  top 10 gains:")
    for g in summary["top10_gains"]:
        print(f"    {g['delta']:+.3f}  {g['image_path']}  ref={g['ref']!r}")
    print("  top 10 losses:")
    for g in summary["top10_losses"]:
        print(f"    {g['delta']:+.3f}  {g['image_path']}  ref={g['ref']!r}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "eval" / "splits_by_document")
    p.add_argument("--baseline-cache", type=Path, default=DEFAULT_BASE_CACHE)
    p.add_argument("--lora-val-csv", type=Path, default=DEFAULT_LORA_VAL)
    p.add_argument("--lora-gt-csv", type=Path, default=DEFAULT_LORA_GT)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--new-gold-target", type=int, default=300)
    p.add_argument("--skip-new-gold", action="store_true")
    p.add_argument("--skip-phash", action="store_true", help="Faster leakage (skip image hashes)")
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = PROJECT_ROOT / "models" / "qwen3-vl-fa-exam-lora-v1" / "training_meta.json"

    # --- 1) Leakage ---
    print("\n" + "#" * 72)
    print("# 1) Leakage audit")
    print("#" * 72)
    if args.skip_phash:
        # Lightweight: composition + document overlap only via run_audit with hashed index
        # Still run full audit — phash is needed by the plan. Keep full.
        pass
    leakage = run_audit(
        args.train,
        args.split_dir,
        ["val", "gold_test"],
        meta_path=meta_path,
    )
    # Also note other manifests' docs
    leakage["all_train_manifest_docs"] = {}
    for mp in TRAIN_MANIFESTS:
        if mp.exists():
            leakage["all_train_manifest_docs"][str(mp)] = composition_report(load_jsonl(mp))
    print_audit(leakage)
    (OUT_DIR / "leakage_audit.json").write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- 2/3) Paired + delta CSV ---
    print("\n" + "#" * 72)
    print("# 2–3) Paired significance + per-line deltas")
    print("#" * 72)
    baseline = load_pred_map_jsonl(args.baseline_cache)
    lora_val_path = args.lora_val_csv if args.lora_val_csv.exists() else FALLBACK_LORA_VAL
    lora_gt_path = args.lora_gt_csv
    if not lora_gt_path.exists():
        raise SystemExit(f"Missing LoRA gold_test errors CSV: {lora_gt_path}")
    lora_val = load_pred_map_csv(lora_val_path)
    lora_gt = load_pred_map_csv(lora_gt_path)

    paired_summaries: dict[str, dict] = {}
    for split, lora_map in (("val", lora_val), ("gold_test", lora_gt)):
        split_rows = load_split_lines(args.split_dir, split)
        rows = build_paired_rows(split_rows, baseline, lora_map)
        missing_b = sum(
            1
            for r in split_rows
            if (r.get("text") or "").strip() and r["image_path"] not in baseline
        )
        missing_l = sum(
            1
            for r in split_rows
            if (r.get("text") or "").strip() and r["image_path"] not in lora_map
        )
        csv_path = OUT_DIR / f"delta_{split}.csv"
        write_delta_csv(rows, csv_path)
        summary = summarize_paired(rows, n_boot=args.n_boot)
        summary["missing_baseline_preds"] = missing_b
        summary["missing_lora_preds"] = missing_l
        summary["delta_csv"] = str(csv_path)
        paired_summaries[split] = summary
        print_paired(split, summary)
        print(f"  wrote {csv_path}")

    (OUT_DIR / "paired_significance.json").write_text(
        json.dumps(paired_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- 4) Held-out document ---
    print("\n" + "#" * 72)
    print("# 4) Held-out-document evaluation")
    print("#" * 72)
    held = heldout_report(args.train, args.split_dir, paired_summaries)
    print(held["note"])
    if held["clean_document_heldout_available"]:
        for split, s in paired_summaries.items():
            print(
                f"  {split}: baseline={s['mean_baseline_sim']:.4f} "
                f"LoRA={s['mean_lora_sim']:.4f} delta={s['mean_delta']:+.4f} "
                f"CI[{s['delta_ci_low']:+.4f},{s['delta_ci_high']:+.4f}] "
                f"sign_p={s['sign_test_p']:.4g}"
            )
    (OUT_DIR / "heldout_document.json").write_text(
        json.dumps(held, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- 5) New gold sealed ---
    new_gold_info: dict = {"skipped": True}
    if not args.skip_new_gold:
        print("\n" + "#" * 72)
        print("# 5) Prepare sealed eval/new_gold")
        print("#" * 72)
        new_gold_info = prepare_new_gold(target=args.new_gold_target)
        print(
            f"  sealed n={new_gold_info['n_sealed']} "
            f"hash_ok={new_gold_info['hash_check_ok']} "
            f"by_notebook={new_gold_info['by_notebook']}"
        )
        print(f"  candidates export: {new_gold_info['candidates_export']}")
        print(f"  out_dir: {new_gold_info['out_dir']}")
    (OUT_DIR / "new_gold_prepare.json").write_text(
        json.dumps(new_gold_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # --- 7) Verdict (OOM doc written separately) ---
    print("\n" + "#" * 72)
    print("# 7) Verdict")
    print("#" * 72)
    val = paired_summaries["val"]
    gt = paired_summaries["gold_test"]
    leak_val = leakage["splits"]["val"]
    leak_gt = leakage["splits"]["gold_test"]

    def leak_serious(audit: dict) -> bool:
        counts = audit.get("overlap_counts") or {}
        return bool(
            counts.get("exact_image_path")
            or counts.get("exact_image_hash")
            or counts.get("source_document")
            or counts.get("source_page")
        )

    leakage_found = leak_serious(leak_val) or leak_serious(leak_gt)
    # Soft text overlap alone is expected for short tokens; call out separately
    soft_text = (leak_val.get("overlap_counts") or {}).get("exact_text", 0) + (
        leak_gt.get("overlap_counts") or {}
    ).get("exact_text", 0)
    soft_near = (leak_val.get("overlap_counts") or {}).get("near_text", 0) + (
        leak_gt.get("overlap_counts") or {}
    ).get("near_text", 0)

    gain_concentrated_gt = gt["top3_positive_gain_share"] >= 0.5
    # Recommendation logic (do not tune on gold_test):
    # Trust val primarily. gold_test is reporting-only.
    if leakage_found:
        recommendation = (
            "REJECT further claims of generalization until leakage is resolved. "
            "Do not train more until manifests are cleaned."
        )
        verdict = "LEAKAGE_BLOCKER"
    elif val["significant_positive"] and val["sign_test_p"] < 0.05:
        recommendation = (
            "LoRA shows a significant gain on held-out val; proceed cautiously. "
            "Treat gold_test as a one-shot report only. Prefer labeling the sealed "
            "new_gold set before more training."
        )
        verdict = "PROMISING_ON_VAL"
    elif val["mean_delta"] < 0 and gt["significant_positive"]:
        recommendation = (
            "Do NOT trust the gold_test jump: val did not improve (and may have "
            "regressed). gold_test n is small and gains may be concentrated. "
            "Pause training; label sealed new_gold; reconsider pseudo-label ratio."
        )
        verdict = "GOLD_TEST_GAIN_UNTRUSTWORTHY"
    else:
        recommendation = (
            "No clear significant win on val. Do not invest in more training until "
            "the sealed new_gold set is labeled and a lower-memory train recipe is ready."
        )
        verdict = "INCONCLUSIVE"

    verdict_table = {
        "leakage_found_serious": leakage_found,
        "soft_text_overlaps_exact": soft_text,
        "soft_text_overlaps_near": soft_near,
        "val_mean_delta": round(val["mean_delta"], 4),
        "val_delta_95ci": [round(val["delta_ci_low"], 4), round(val["delta_ci_high"], 4)],
        "val_sign_test_p": val["sign_test_p"],
        "val_significant_positive": val["significant_positive"],
        "gold_test_mean_delta": round(gt["mean_delta"], 4),
        "gold_test_delta_95ci": [round(gt["delta_ci_low"], 4), round(gt["delta_ci_high"], 4)],
        "gold_test_sign_test_p": gt["sign_test_p"],
        "gold_test_significant_positive": gt["significant_positive"],
        "gold_test_gain_concentrated_top3": gain_concentrated_gt,
        "gold_test_top3_positive_gain_share": gt["top3_positive_gain_share"],
        "heldout_document_available": held["clean_document_heldout_available"],
        "heldout_documents": held["heldout_documents"],
        "train_documents": held["train_documents"],
        "verdict": verdict,
        "recommendation": recommendation,
    }

    print(
        f"{'question':<36} {'answer'}\n"
        f"{'-'*36} {'-'*40}\n"
        f"{'leakage (image/doc/page)?':<36} {leakage_found}\n"
        f"{'soft text overlaps (exact/near)?':<36} {soft_text}/{soft_near}\n"
        f"{'val delta significant (+)?':<36} {val['significant_positive']} "
        f"(Δ={val['mean_delta']:+.4f}, CI=[{val['delta_ci_low']:+.4f},{val['delta_ci_high']:+.4f}], "
        f"p={val['sign_test_p']:.4g})\n"
        f"{'gold_test delta significant (+)?':<36} {gt['significant_positive']} "
        f"(Δ={gt['mean_delta']:+.4f}, p={gt['sign_test_p']:.4g}) "
        f"[report-only, not for tuning]\n"
        f"{'gold_test gain concentrated?':<36} {gain_concentrated_gt} "
        f"(top3 share={gt['top3_positive_gain_share']:.1%})\n"
        f"{'held-out document available?':<36} {held['clean_document_heldout_available']} "
        f"{held['heldout_documents']}\n"
        f"{'verdict':<36} {verdict}\n"
    )
    print(f"RECOMMENDATION:\n  {recommendation}")

    (OUT_DIR / "verdict.json").write_text(
        json.dumps(verdict_table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nAll artifacts under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
