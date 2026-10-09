#!/usr/bin/env python3
"""Score local models against hand corrections on sealed new_gold.

Only corrected (non-skip, non-illegible-empty optional) lines are scored.
Reports body vs form_field separately, model agreement, repeat consistency,
and blind vs pre-filled differences.

Usage:
  python -m eval.new_gold_report
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import bootstrap_mean_ci, cer, sentence_similarity, wer  # noqa: E402
from eval.normalize import normalize_text  # noqa: E402
from labeling.db import DEFAULT_DB, connect, latest_non_skip  # noqa: E402
from labeling.queue import DEFAULT_QUEUE  # noqa: E402

MODEL_IDS = ("qwen3-vl-2b-persian", "qwen2.5-vl-7b")


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _score_pairs(refs: list[str], preds: list[str], *, n_boot: int = 1000) -> dict:
    sims = [sentence_similarity(p, r) for p, r in zip(preds, refs)]
    if not sims:
        return {"n": 0}
    ref_j = "\n".join(normalize_text(r) for r in refs)
    pred_j = "\n".join(normalize_text(p) for p in preds)
    boot = bootstrap_mean_ci(sims, n_boot=n_boot)
    return {
        "n": len(sims),
        "mean_sentence_similarity": round(sum(sims) / len(sims), 6),
        "bootstrap_95ci": {
            "low": round(boot["ci_low"], 6),
            "high": round(boot["ci_high"], 6),
            "n_boot": n_boot,
        },
        "pct_similarity_ge_0_80": round(sum(1 for s in sims if s >= 0.8) / len(sims), 6),
        "cer": round(cer(ref_j, pred_j), 6),
        "wer": round(wer(ref_j, pred_j), 6),
    }


def load_preds(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for r in conn.execute("SELECT line_id, model_id, text FROM model_preds"):
        out[r["line_id"]][r["model_id"]] = r["text"]
    return out


def load_scored_corrections(conn: sqlite3.Connection, queue: str) -> list[dict]:
    latest = latest_non_skip(conn, queue)
    rows = []
    for lid, r in latest.items():
        if (r.get("notes") or "") == "__UNDO__":
            continue
        action = r.get("action") or ""
        if action == "skipped":
            continue
        text = r.get("corrected_text") or ""
        if action == "marked_illegible":
            # Include as empty ref only if we want — skip from similarity metrics
            continue
        if not text.strip():
            continue
        rows.append(dict(r))
    return rows


def repeat_consistency(conn: sqlite3.Connection, queue: str) -> dict:
    """Agreement on covert repeats: same line_id labeled more than once (non-undo)."""
    rows = list(
        conn.execute(
            """
            SELECT line_id, corrected_text, action, is_repeat, id, notes
            FROM corrections
            WHERE queue=? AND notes != '__UNDO__' AND action IN ('edited','accepted_as_is','marked_illegible')
            ORDER BY id ASC
            """,
            (queue,),
        )
    )
    by_line: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        key = normalize_text(r["corrected_text"] or "")
        if r["action"] == "marked_illegible":
            key = "__ILLEGIBLE__"
        by_line[r["line_id"]].append(key)
    pairs = 0
    agree = 0
    for lid, texts in by_line.items():
        if len(texts) < 2:
            continue
        # Compare first vs each later
        first = texts[0]
        for t in texts[1:]:
            pairs += 1
            if t == first:
                agree += 1
    return {
        "n_repeat_pairs": pairs,
        "n_agree": agree,
        "agreement_rate": round(agree / pairs, 6) if pairs else None,
    }


def blind_vs_prefilled(rows: list[dict], preds: dict[str, dict[str, str]], model_id: str) -> dict:
    groups = {"blind": [], "prefilled": []}
    for r in rows:
        key = "blind" if int(r.get("blind_mode") or 0) else "prefilled"
        pred = preds.get(r["line_id"], {}).get(model_id)
        if pred is None:
            continue
        groups[key].append((r["corrected_text"], pred))
    out = {}
    for g, pairs in groups.items():
        if not pairs:
            out[g] = {"n": 0}
            continue
        refs, hyps = zip(*pairs)
        out[g] = _score_pairs(list(refs), list(hyps))
    return out


def build_report(
    *,
    db_path: Path = DEFAULT_DB,
    queue: str = DEFAULT_QUEUE,
    n_boot: int = 1000,
) -> dict:
    conn = connect(db_path)
    rows = load_scored_corrections(conn, queue)
    preds = load_preds(conn)

    by_region: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_region[r.get("region_type") or "body"].append(r)
        by_region["all"].append(r)

    models_report = {}
    for mid in MODEL_IDS:
        models_report[mid] = {}
        for region, subset in by_region.items():
            refs, hyps = [], []
            for r in subset:
                p = preds.get(r["line_id"], {}).get(mid)
                if p is None:
                    continue
                refs.append(r["corrected_text"])
                hyps.append(p)
            models_report[mid][region] = _score_pairs(refs, hyps, n_boot=n_boot)
        models_report[mid]["blind_vs_prefilled"] = blind_vs_prefilled(rows, preds, mid)

    # Inter-model agreement + correlation with true similarity (on shared lines)
    both_refs, both_a, both_b, sim_a, sim_b, inter = [], [], [], [], [], []
    for r in rows:
        pa = preds.get(r["line_id"], {}).get(MODEL_IDS[0])
        pb = preds.get(r["line_id"], {}).get(MODEL_IDS[1])
        if pa is None or pb is None:
            continue
        ref = r["corrected_text"]
        both_refs.append(ref)
        both_a.append(pa)
        both_b.append(pb)
        sa = sentence_similarity(pa, ref)
        sb = sentence_similarity(pb, ref)
        sim_a.append(sa)
        sim_b.append(sb)
        inter.append(sentence_similarity(pa, pb))

    agreement = {
        "n_shared": len(inter),
        "mean_inter_model_similarity": round(sum(inter) / len(inter), 6) if inter else None,
        "corr_inter_vs_true_qwen3": round(_pearson(inter, sim_a), 6) if inter else None,
        "corr_inter_vs_true_qwen25": round(_pearson(inter, sim_b), 6) if inter else None,
        "corr_qwen3_sim_vs_qwen25_sim": round(_pearson(sim_a, sim_b), 6) if sim_a else None,
    }

    return {
        "queue": queue,
        "n_corrected_scored": len(rows),
        "n_by_region": {k: len(v) for k, v in by_region.items()},
        "models": models_report,
        "model_agreement": agreement,
        "reviewer_repeat_consistency": repeat_consistency(conn, queue),
        "sealed": True,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--queue", default=DEFAULT_QUEUE)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval" / "new_gold_report.json",
    )
    args = p.parse_args(argv)

    if not args.db.exists():
        print(f"No DB at {args.db}. Launch the UI or run precompute first.")
        return 1
    report = build_report(db_path=args.db, queue=args.queue, n_boot=args.n_boot)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if report["n_corrected_scored"] == 0:
        print("\n(No corrections yet — label via: python -m labeling.ui --queue new_gold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
