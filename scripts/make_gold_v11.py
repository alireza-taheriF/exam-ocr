#!/usr/bin/env python3
"""Build dataset/gold_v1.1.jsonl: gold.jsonl with keyboard-layout-garbled rows repaired.

gold.jsonl itself is never modified. Only rows whose text matches a known
Latin-typed-on-Persian-layout pattern are rewritten, using the deterministic
standard Persian (ISO 9995 / Windows) QWERTY layout, unshifted keys. Every change
is logged to dataset/gold_v1.1_changes.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FA = dict(zip("qwertyuiop[]asdfghjkl;'zxcvbnm,",
              "ضصثقفغعهخحجچشسیبلاتنمکگظطزرذدپو"))


def qwerty_to_fa(s: str) -> str:
    return "".join(FA.get(c, c) for c in s)


# row-id (index in gold.jsonl) -> expected garbled prefix, as a safety check
TARGETS = {201: "fun hc iv ]kn"}
SKIPPED = {200: "Persian-layout keys typed for Latin text ('CamScanner' -> 'شپئزشددثق'); "
                "the inverse mapping cannot recover the leading 'C' (shift+c), so the "
                "row is left unchanged rather than replaced with a known-wrong label."}


def main():
    rows = [json.loads(l) for l in (ROOT / "dataset/gold.jsonl").open(encoding="utf-8") if l.strip()]
    log = []
    for i, new_text_fn in TARGETS.items():
        r = rows[i]
        assert r["text"].startswith(TARGETS[i]), (i, r["text"])
        new = qwerty_to_fa(r["text"])
        log.append({"row": i, "image_path": r["image_path"], "old": r["text"], "new": new})
        r["text"] = new
        r["gold_v11_fixed"] = True
    with (ROOT / "dataset/gold_v1.1.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (ROOT / "dataset/gold_v1.1_changes.json").write_text(
        json.dumps({"changed": log, "skipped": {str(k): v for k, v in SKIPPED.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(log, ensure_ascii=False, indent=2))


main()
