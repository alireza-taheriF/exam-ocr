"""Deep learning loop: student corrections → new LoRA → eval gate → promote or reject.

    .venv/bin/python -m learning.retrain status
    .venv/bin/python -m learning.retrain prepare            # → outputs/colab/exam_ocr_lora_data.zip
    # train on Colab with training/lora/train_qwen_lora_colab.ipynb, unzip the adapter into models/<label>/
    .venv-qwen/bin/python -m learning.retrain evaluate --label lora_v3 --adapter models/lora_v3
    .venv/bin/python -m learning.retrain rollback            # previous champion goes live again

Training itself runs on Colab: this 16 GB Mac ran out of memory during a LoRA run
(training/lora/README.md). Everything around it — which corrections are usable,
the data bundle, scoring and promotion — runs here and follows
`.claude/skills/feedback-learning` and `.claude/skills/ocr-eval-gate`:

- only trusted corrections (``learning/label_trust.trust_for``; an admin verdict
  overrides the heuristics) from confirmed, consented, non-hold-out documents
  are trained on, mixed with the public handwriting lines as replay;
- a new adapter goes live only if it beats the champion on gold (eval_gate's
  promotion rule) **and** is not worse on the correction hold-out documents.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from learning import registry  # noqa: E402
from learning.label_trust import TRUSTED, trust_for  # noqa: E402

HW_MANIFESTS = [PROJECT_ROOT / "dataset" / "external" / "hw_lines" / f"{s}.jsonl" for s in ("train", "validation")]
DEFAULT_BUNDLE = PROJECT_ROOT / "outputs" / "colab" / "exam_ocr_lora_data.zip"
EXPORTS = PROJECT_ROOT / "exports"


def min_new_corrections() -> int:
    return int(os.environ.get("RETRAIN_MIN_CORRECTIONS", "500"))


def state_path() -> Path:
    return registry.registry_path().with_name("retrain_state.json")


def load_state() -> dict:
    path = state_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _usable(row: dict) -> bool:
    return bool(row["consent"]) and trust_for(row).status == TRUSTED


def status(store) -> dict[str, Any]:
    rows = store.final_pairs(confirmed_only=True)
    usable = [r for r in rows if _usable(r)]
    train = sum(1 for r in usable if not r["holdout"])
    holdout = sum(1 for r in usable if r["holdout"])
    champion = registry.load().get("champion") or {}
    used = int(champion.get("corrections_used") or 0)
    new = max(0, train - used)
    threshold = min_new_corrections()
    return {
        "champion": champion.get("label"),
        "champion_accuracy_pct": champion.get("accuracy_pct"),
        "trusted_train_corrections": train,
        "trusted_holdout_corrections": holdout,
        "corrections_in_champion": used,
        "new_corrections": new,
        "threshold": threshold,
        "ready": new >= threshold,
        "prepared": load_state().get("prepared"),
        "handwriting_lines_ready": all(m.exists() for m in HW_MANIFESTS),
    }


def prepare(store, *, out: Path = DEFAULT_BUNDLE, include_handwriting: bool = True) -> dict[str, Any]:
    """Export trusted corrections and zip them with the handwriting lines for Colab."""
    from scripts.export_corrections import export
    from training.lora.make_bundle import build

    stamp = time.strftime("%Y%m%d_%H%M%S")
    export_dir = EXPORTS / f"corrections_{stamp}"
    counts = export(store, export_dir)
    manifests = []
    if counts.get("train"):
        manifests.append(export_dir / "train.jsonl")
    if include_handwriting:
        missing = [m for m in HW_MANIFESTS if not m.exists()]
        if missing:
            raise FileNotFoundError(f"handwriting lines missing ({missing[0]}); run scripts/build_hw_lines.py")
        manifests += HW_MANIFESTS
    if not manifests:
        raise ValueError("nothing to train on: no trusted corrections and handwriting lines disabled")
    summary = build(manifests, out)
    prepared = {
        "bundle": str(out),
        "export_dir": str(export_dir),
        "corrections_used": int(counts.get("train", 0)),
        "holdout_corrections": int(counts.get("holdout", 0)),
        "manifests": summary,
        "created_at": time.time(),
    }
    state = load_state()
    state["prepared"] = prepared
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return prepared


def holdout_check(export_dir: Path, adapter: str) -> dict[str, Any]:
    """CER of the candidate vs. the model that originally served each hold-out line."""
    from PIL import Image

    from pipeline.engines import QwenLocalEngine
    from training.eval.metrics import corpus_cer

    manifest = export_dir / "holdout.jsonl"
    rows = [json.loads(l) for l in manifest.open(encoding="utf-8") if l.strip()] if manifest.exists() else []
    if not rows:
        return {"lines": 0, "passed": True, "note": "no hold-out corrections yet"}
    engine = QwenLocalEngine(adapter=adapter)
    preds = [engine.recognize(Image.open(export_dir / r["image_path"]).convert("RGB"))[0] for r in rows]
    refs = [r["text"] for r in rows]
    cand = corpus_cer(refs, preds)
    served = corpus_cer(refs, [r.get("raw_text") or r["predicted_text"] for r in rows])
    return {"lines": len(rows), "candidate_cer": round(cand, 4), "served_cer": round(served, 4),
            "served_versions": sorted({r["model_version"] for r in rows}), "passed": cand <= served}


def evaluate(label: str, adapter: str, *, python: str = sys.executable) -> dict[str, Any]:
    """Gold gate (registers champion/rejected), then the correction hold-out check."""
    before = (registry.load().get("champion") or {}).get("label")
    cmd = [python, str(PROJECT_ROOT / "scripts" / "eval_gate.py"), "--label", label, "--adapter", adapter, "--register"]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    reg = registry.load()
    champion = reg.get("champion") or {}
    result: dict[str, Any] = {"label": label, "gold_gate": "champion" if champion.get("label") == label else "rejected"}
    if result["gold_gate"] != "champion":
        return result

    prepared = load_state().get("prepared") or {}
    hold = holdout_check(Path(prepared["export_dir"]), adapter) if prepared.get("export_dir") else \
        {"lines": 0, "passed": True, "note": "no prepared export recorded"}
    result["holdout"] = hold
    reg = registry.load()
    if not hold["passed"]:
        # Gold improved but real students' sheets got worse: keep the old model live.
        for e in reversed(reg["history"]):
            if e.get("label") == label:
                e["status"] = "rejected"
                e["reject_reason"] = "correction_holdout_regressed"
                e["holdout"] = hold
                break
        registry.save(reg)
        if before:
            registry.activate(before, reason=f"{label} regressed on correction hold-out")
        result["status"] = "rejected"
        return result

    reg["champion"]["corrections_used"] = prepared.get("corrections_used", 0)
    reg["champion"]["holdout"] = hold
    for e in reversed(reg["history"]):
        if e.get("label") == label:
            e.update(corrections_used=prepared.get("corrections_used", 0), holdout=hold)
            break
    registry.save(reg)
    result["status"] = "champion"
    return result


def _store():
    from app.store import Store

    db = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")) / "exam_ocr.sqlite3"
    return Store(db)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pp = sub.add_parser("prepare")
    pp.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    pp.add_argument("--no-handwriting", action="store_true")
    pe = sub.add_parser("evaluate")
    pe.add_argument("--label", required=True)
    pe.add_argument("--adapter", required=True)
    pa = sub.add_parser("activate")
    pa.add_argument("label")
    sub.add_parser("rollback")
    args = p.parse_args(argv)

    if args.cmd == "status":
        out = status(_store())
    elif args.cmd == "prepare":
        out = prepare(_store(), out=args.out, include_handwriting=not args.no_handwriting)
    elif args.cmd == "evaluate":
        out = evaluate(args.label, args.adapter)
    elif args.cmd == "activate":
        out = registry.activate(args.label, reason="manual")
    else:
        prev = registry.previous_champion()
        if prev is None:
            print("no previous champion to roll back to")
            return 1
        out = registry.activate(prev, reason="rollback")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    if args.cmd in {"activate", "rollback", "evaluate"}:
        print("Restart the server (or POST /api/admin/reload-engine) for the change to take effect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
