#!/usr/bin/env python3
"""Hand-transcribe a gold set, one crop at a time.

This is the only step in the project that produces ground truth. Every label in
`dataset/` is pseudo-label output from an auto-labeler, so CER computed against it
measures agreement with that labeler, not accuracy. Until this file exists there
is no way to accept or reject a labeling backend — including a paid one.

Sampling (per `training/eval/eval_gold_subset.py`'s recipe):
  * drawn from the TEST split only, so the gold set stays held out from training
    and from backend selection;
  * restricted to documents with >= 50 lines. Test has 60 groups but 56 are
    single-page crawled web images contributing 1-3 lines each — sampling evenly
    across all 60 would spend most of the typing budget on web screenshots
    rather than exam handwriting.

Every entry is appended and flushed immediately, so an interrupt costs at most
the line being typed. `--resume` skips crops already in the output.

Usage:
    python scripts/label_gold_interactive.py                    # 240 lines, resume
    python scripts/label_gold_interactive.py --limit 40         # a short session
    python scripts/label_gold_interactive.py --no-open          # don't open images

At each prompt:
    <text>   the transcription
    [Enter]  illegible — records an empty label, which the scorer skips
    s        skip without recording, so it comes back next session
    u        undo the previous entry
    q        save and quit
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

DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "test_v3.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "dataset" / "gold.jsonl"
SUBSTANTIVE_MIN_LINES = 50
DEFAULT_TARGET = 240

HELP = """
  Type what you actually see in the crop, then press Enter.

    <text>   the transcription
    [Enter]  illegible — recorded with an empty label and skipped when scoring
    s        skip, without recording (offered again next session)
    u        undo the previous entry
    q        save and quit
    ?        show this help

  Transcribe what is written, not what it should say: keep the writer's spelling,
  their digits, and their mistakes. Normalization happens later and is reversible;
  a silently corrected label is not.
"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_gold_todo(rows: list[dict], *, target: int, seed: int) -> list[dict]:
    """Even draw across the substantial documents, largest-document-first order."""
    by_doc: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_doc[row.get("source_document", "unknown")].append(row)

    docs = {d: rs for d, rs in by_doc.items() if len(rs) >= SUBSTANTIVE_MIN_LINES}
    if not docs:  # tiny corpus — fall back to everything rather than sampling nothing
        docs = dict(by_doc)

    rng = random.Random(seed)
    per_doc = max(1, target // len(docs))
    picked: list[dict] = []
    for doc in sorted(docs, key=lambda d: (-len(docs[d]), d)):
        pool = docs[doc]
        picked += rng.sample(pool, min(per_doc, len(pool)))
    rng.shuffle(picked)   # interleave documents so fatigue is spread evenly
    return picked


def show(path: Path, *, open_image: bool) -> None:
    if not open_image:
        return
    try:
        # -g keeps focus in the terminal, so typing continues uninterrupted.
        subprocess.run(["open", "-g", str(path)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Hand-transcribe a gold set")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help=f"Split to sample from (default {DEFAULT_SOURCE.name} — the "
                        f"test split, so gold stays held out)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET,
                   help=f"Gold-set size to sample (default {DEFAULT_TARGET})")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N entries this session (0 = no limit)")
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed. Keep it fixed across sessions or --resume "
                        "will be comparing against a different sample.")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-offer crops that already have a label")
    p.add_argument("--no-open", action="store_true",
                   help="Print the path instead of opening the image")
    args = p.parse_args()

    if not args.source.exists():
        print(f"Not found: {args.source}")
        print("Run: python scripts/split_dataset.py --input dataset/lines_v3.jsonl")
        return 1

    todo = sample_gold_todo(load_jsonl(args.source), target=args.target,
                            seed=args.seed)
    if not todo:
        print(f"{args.source} yielded no crops to label.")
        return 1

    done: dict[str, str] = {}
    if args.out.exists() and not args.no_resume:
        done = {r["image_path"]: r.get("text", "") for r in load_jsonl(args.out)}
    remaining = [r for r in todo if r["image_path"] not in done]

    docs = sorted({r["source_document"] for r in todo})
    print(f"\n  gold set    : {args.out}")
    print(f"  sampled     : {len(todo)} crops across {len(docs)} document(s)")
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
                remaining.insert(i, row)   # re-offer this crop
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
                # Rewrite without the last entry: an append-only file cannot
                # retract a line in place, and leaving a wrong label in is worse
                # than the cost of a rewrite.
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
                    # A human typing a markdown table or an emoji is almost
                    # always a paste accident. Confirm rather than silently
                    # accepting a label the ingest path would reject.
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
            sink.flush()          # survive an interrupt at any point
            written += 1
            session.append({**entry, "_row": row})
            if answer:
                print(f"      saved: {normalize_text(answer)[:60]}")
            else:
                print("      saved as illegible (empty label)")

    labelled = load_jsonl(args.out) if args.out.exists() else []
    non_empty = [r for r in labelled if normalize_text(r.get("text") or "")]
    docs_covered = {r["source_document"] for r in non_empty}
    print(f"\n  this session : {written} entries")
    print(f"  gold total   : {len(labelled)} rows, {len(non_empty)} with text, "
          f"{len(docs_covered)} document(s)")
    print(f"  file         : {args.out}")
    if non_empty:
        print("\n  Score any prediction set against it:")
        print(f"      python training/eval/eval_gold_subset.py --gold {args.out} "
              f"--pred predictions.jsonl --by-document")
    if len(non_empty) < 100:
        print(f"\n  {len(non_empty)} labelled lines is thin for a CER estimate — "
              f"200+ makes the number stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
