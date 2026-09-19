#!/usr/bin/env python3
"""Throwaway interactive labels for a Qwen2.5-VL GGUF CER check.

Not the gold set. Samples padded Phase-1B crops from `dataset/lines_v3.jsonl`
across handwritten source documents (skips 1–3-line web screenshots), shows
each crop, and appends to `dataset/qwen_test_sample.jsonl`.

Usage:
    python scripts/label_qwen_test_sample.py
    python scripts/label_qwen_test_sample.py --limit 10
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalizer import normalize_text, vlm_artifact_reason  # noqa: E402

DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "lines_v3.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "qwen_test_sample.jsonl"
SUBSTANTIVE_MIN_LINES = 50
DEFAULT_TARGET = 30

HELP = """
  Type what you actually see in the crop, then press Enter.

    <text>   the transcription
    [Enter]  illegible — recorded empty (skipped when scoring)
    s        skip, without recording
    u        undo the previous entry this session
    q        save and quit
    ?        show this help
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_qwen_todo(rows: list[dict], *, target: int, seed: int) -> list[dict]:
    """Even draw across substantial documents, then fill to `target`."""
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_doc[row.get("source_document", "unknown")].append(row)

    docs = {d: rs for d, rs in by_doc.items() if len(rs) >= SUBSTANTIVE_MIN_LINES}
    if not docs:
        docs = dict(by_doc)

    rng = random.Random(seed)
    n_docs = len(docs)
    per_doc = max(1, target // n_docs)
    picked: list[dict] = []
    used: set[str] = set()
    for doc in sorted(docs, key=lambda d: (-len(docs[d]), d)):
        pool = docs[doc]
        take = rng.sample(pool, min(per_doc, len(pool)))
        for row in take:
            picked.append(row)
            used.add(row["image_path"])

    if len(picked) < target:
        leftover = [r for r in rows if r["image_path"] not in used
                    and r.get("source_document", "unknown") in docs]
        rng.shuffle(leftover)
        for row in leftover[: target - len(picked)]:
            picked.append(row)
            used.add(row["image_path"])

    rng.shuffle(picked)
    return picked[:target]


def show(path: Path, *, open_image: bool) -> None:
    if not open_image:
        return
    try:
        subprocess.run(["open", "-g", str(path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Hand-transcribe a Qwen GGUF test sample")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    if not args.source.exists():
        print(f"Not found: {args.source}")
        return 1

    todo = sample_qwen_todo(load_jsonl(args.source), target=args.target, seed=args.seed)
    if not todo:
        print(f"{args.source} yielded no crops to label.")
        return 1

    done: dict[str, str] = {}
    if args.out.exists() and not args.no_resume:
        done = {r["image_path"]: r.get("text", "") for r in load_jsonl(args.out)}
    remaining = [r for r in todo if r["image_path"] not in done]

    docs = sorted({r["source_document"] for r in todo})
    print(f"\n  sample      : {args.out}")
    print(f"  sampled     : {len(todo)} crops across {len(docs)} document(s)")
    for d in docs:
        n = sum(1 for r in todo if r["source_document"] == d)
        print(f"                {n:2d}  {d}")
    if done:
        print(f"  already done: {len(done)}  ({len(remaining)} left)")
    if args.limit:
        print(f"  this session: up to {args.limit}")
    print(HELP)

    if not remaining:
        print("  Nothing left to label — the sampled set is complete.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    session: list[dict] = []
    with args.out.open("a", encoding="utf-8") as sink:
        for i, row in enumerate(remaining, 1):
            if args.limit and written >= args.limit:
                print(f"\n  Session limit reached ({args.limit}).")
                break

            crop = PROJECT_ROOT / row["image_path"]
            if not crop.exists():
                print(f"  [{i}/{len(remaining)}] MISSING {row['image_path']} — skipped")
                continue

            print(f"\n  [{i}/{len(remaining)}]  {row['image_path']}")
            print(f"      doc {row['source_document']}  "
                  f"page {Path(row['source_image']).name}  line {row['line_index']}")
            if args.no_open:
                print(f"      open: {crop}")
            show(crop, open_image=not args.no_open)

            try:
                answer = input("      > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Interrupted — everything already entered is saved.")
                break

            if answer == "?":
                print(HELP)
                remaining.insert(i, row)
                continue
            if answer == "q":
                print("\n  Saved and stopping.")
                break
            if answer == "s":
                continue
            if answer == "u":
                if not session:
                    print("      nothing to undo this session")
                    continue
                dropped = session.pop()
                sink.flush()
                kept = [r for r in load_jsonl(args.out)
                        if r["image_path"] != dropped["image_path"]]
                args.out.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                    encoding="utf-8",
                )
                written -= 1
                print(f"      undone: {dropped['image_path']}")
                remaining.insert(i, dropped["_row"])
                continue

            if answer:
                reason = vlm_artifact_reason(answer)
                if reason:
                    print(f"      NOTE: this looks like a transcription artifact "
                          f"({reason}).")
                    if input("      keep it anyway? [y/N] ").strip().lower() != "y":
                        remaining.insert(i, row)
                        continue

            entry = {
                "image_path": row["image_path"],
                "text": answer,
                "source_document": row["source_document"],
                "line_index": row["line_index"],
                "source_image": row["source_image"],
            }
            sink.write(json.dumps(entry, ensure_ascii=False) + "\n")
            sink.flush()
            written += 1
            session.append({**entry, "_row": row})
            if answer:
                print(f"      saved: {normalize_text(answer)[:60]}")
            else:
                print("      saved as illegible (empty label)")

    labelled = load_jsonl(args.out) if args.out.exists() else []
    non_empty = [r for r in labelled if normalize_text(r.get("text") or "")]
    print(f"\n  this session : {written} entries")
    print(f"  sample total : {len(labelled)} rows, {len(non_empty)} with text")
    print(f"  file         : {args.out}")
    print("\n  When done, score with:")
    print("      python scripts/eval_qwen_gguf_sample.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
