#!/usr/bin/env python3
"""Append labels read from one contact sheet to a jsonl (resumable, deduped).

    add_labels.py --sheet-json S/sheet_000.json --todo dataset/dev_todo.jsonl \
        --out dataset/dev_claude.jsonl <<< '{"1":"متن","2":null}'

stdin is a JSON object index -> text; null / "" marks the crop illegible.
"""
import argparse, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--sheet-json", type=Path, required=True)
p.add_argument("--todo", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--labeler", default="claude-code")
a = p.parse_args()

mapping = json.loads(a.sheet_json.read_text())
meta = {}
for l in a.todo.open(encoding="utf-8"):
    if l.strip():
        r = json.loads(l); meta[r["image_path"]] = r
labels = json.loads(sys.stdin.read())
have = set()
if a.out.exists():
    have = {json.loads(l)["image_path"] for l in a.out.open(encoding="utf-8") if l.strip()}
missing = set(mapping) - set(labels)
if missing:
    sys.exit(f"missing indices: {sorted(missing, key=int)}")
n = 0
with a.out.open("a", encoding="utf-8") as f:
    for idx, path in mapping.items():
        if path in have:
            continue
        t = (labels[idx] or "").strip()
        f.write(json.dumps({"image_path": path,
                            "source_document": meta[path]["source_document"],
                            "text": t, "illegible": not t,
                            "labeler": a.labeler}, ensure_ascii=False) + "\n")
        n += 1
print(f"added {n}, total {len(have) + n}")
