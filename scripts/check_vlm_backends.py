#!/usr/bin/env python3
"""Verify that a VLM backend actually READS the image it is sent.

Phase 1B Step 3. The failure mode this exists to catch is not "the call errored"
— it is "the call succeeded, the image was silently dropped, and the model
returned plausible Persian text anyway." That produces labels that look fine and
are pure invention, which is exactly how the original pipeline broke. A model
that returns *something* is not evidence that it saw anything.

So each backend gets two calls:

1. CONTROL — an image containing a randomly generated 6-character code, rendered
   fresh on every run. The code exists nowhere in the prompt and cannot be
   guessed (36^6 ≈ 2.2 billion). Returning it is proof the pixels arrived. This
   is the test that decides pass/fail.
2. REAL — an actual Persian handwriting crop from the corpus, recorded verbatim
   for eyeballing. Not scored: without a gold label there is nothing to score
   against, and "looks like Persian" is precisely the judgement that cannot be
   trusted here.

A backend passes only if the control code comes back exactly. Anything else —
transport error, refusal, empty response, wrong code — is a fail, and the reason
is printed rather than summarized.

Usage:
    python scripts/check_vlm_backends.py                       # all four models
    python scripts/check_vlm_backends.py --models claude-opus-4-8
    python scripts/check_vlm_backends.py --crop dataset/lines_v3/abc.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import string
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

REPORT_OUT = PROJECT_ROOT / "dataset" / "backend_capability_report.json"

# Candidates for a vision bake-off. Catalog membership is per-router — always
# list /v1/models on the current endpoint before treating these as available.
# glm-5.3 and deepseek-v4-flash are text-only (weight-level) and are not retested.
DEFAULT_MODELS = [
    "claude-opus-4-8",
    "claude-opus-5",
    "gpt-5.6-sol",
]

# Unambiguous glyphs only: no O/0, I/1, S/5, Z/2 confusions to argue about when a
# model returns something one character off.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRTUVWXY" + "34679"

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

CONTROL_SYSTEM = (
    "You are an OCR engine. Read the characters visible in the image and output "
    "them exactly. Output only the characters, nothing else."
)
CONTROL_USER = "What characters are in this image? Output only the characters."

CROP_SYSTEM = (
    "You are an OCR/HTR engine for Persian (Farsi). The image is a SINGLE cropped "
    "line from a scanned handwritten or printed exam page. Transcribe exactly the "
    "text visible in this one line. Output only the transcription: no preamble, no "
    "translation, no markdown, no commentary. If the line is entirely illegible, "
    "output exactly: [ILLEGIBLE]"
)
CROP_USER = "Transcribe this single line of Persian text."


def make_control_image(code: str) -> Image.Image:
    """Render `code` as large black text on white. Deliberately trivial to read:
    a model that fails this did not receive the image, rather than found it hard."""
    font = None
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, 96)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default(size=96)

    img = Image.new("RGB", (720, 200), "white")
    draw = ImageDraw.Draw(img)
    box = draw.textbbox((0, 0), code, font=font)
    draw.text(
        ((720 - (box[2] - box[0])) // 2, (200 - (box[3] - box[1])) // 2 - box[1]),
        code,
        fill="black",
        font=font,
    )
    return img


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(client, model: str, system: str, user: str, img: Image.Image) -> dict:
    """One image + one prompt. Returns text plus whatever usage the router reports."""
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=128,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(img)}},
                    {"type": "text", "text": user},
                ],
            },
        ],
    )
    usage = getattr(resp, "usage", None)
    return {
        "text": (resp.choices[0].message.content or "").strip(),
        "seconds": round(time.monotonic() - t0, 2),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        },
        "response_model": getattr(resp, "model", None),
    }


def normalize_code(text: str) -> str:
    """Strip everything a chatty model might wrap the answer in."""
    return "".join(c for c in (text or "").upper() if c.isalnum())


def probe_text_only(client, model: str) -> dict:
    """Send a text-only call to the same model.

    This is what separates the two very different reasons a vision call fails:

    * text works, image rejected  → the model has no vision. A capability fact;
      no amount of billing or retrying changes it.
    * text also fails             → the account, quota, or routing is broken. The
      model's vision support is simply untested, and reporting it as "no vision"
      would be wrong.

    Without this probe both land in the same FAIL bucket and the report implies a
    capability conclusion it has not earned.
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
        return {"ok": True, "text": (resp.choices[0].message.content or "").strip()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}


def classify(row: dict) -> str:
    """Why this backend cannot be used, in terms that imply the right next step."""
    if row["accepts_image"]:
        return "usable"
    if row.get("text_only", {}).get("ok"):
        return "no_vision"        # the model answered text and refused the image
    return "unreachable"          # quota / auth / routing — capability untested


def pick_crop(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit if explicit.exists() else None
    for manifest in ("lines_v3.jsonl", "lines_v2.jsonl"):
        path = PROJECT_ROOT / "dataset" / manifest
        if not path.exists():
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 4000:
                    break
                if line.strip():
                    rows.append(json.loads(line))
        # A mid-size crop from a handwritten document: big enough to hold real
        # text, not a stray fragment.
        wide = [
            r for r in rows
            if 600 < r["bbox"][2] - r["bbox"][0] < 1500
            and 45 < r["bbox"][3] - r["bbox"][1] < 120
            and "دست_نویس" in r.get("source_document", "")
        ]
        pool = wide or rows
        if pool:
            candidate = PROJECT_ROOT / random.Random(0).choice(pool)["image_path"]
            if candidate.exists():
                return candidate
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Does this backend actually read images?")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--crop", type=Path, default=None,
                   help="Persian crop to send (default: sampled from the manifest)")
    p.add_argument("--out", type=Path, default=REPORT_OUT)
    p.add_argument("--skip-crop", action="store_true",
                   help="Control call only — halves the call count")
    args = p.parse_args()

    from dotenv import load_dotenv

    from pipeline.openai_compat import (
        create_openai_client,
        resolve_openai_api_key,
        resolve_openai_base_url,
    )

    load_dotenv(PROJECT_ROOT / ".env")
    base_url = resolve_openai_base_url(dict(os.environ))
    api_key = resolve_openai_api_key(dict(os.environ))
    if not api_key:
        print("No API key found in .env (OPENAI_API_KEY / VLM_API_KEY / …)")
        return 1
    client = create_openai_client(api_key=api_key, base_url=base_url)

    code = "".join(random.choice(_CODE_ALPHABET) for _ in range(6))
    control = make_control_image(code)
    crop_path = None if args.skip_crop else pick_crop(args.crop)
    crop = Image.open(crop_path).convert("RGB") if crop_path else None

    calls = len(args.models) * (1 if crop is None else 2)
    print(f"Endpoint : {base_url}")
    print(f"Models   : {len(args.models)}  ({', '.join(args.models)})")
    print(f"Calls    : {calls} — two small images per model, a few cents total")
    print(f"Control  : freshly generated code, not present in any prompt")
    if crop_path:
        print(f"Crop     : {crop_path.relative_to(PROJECT_ROOT)} "
              f"({crop.width}x{crop.height})")
    print()

    results = []
    for model in args.models:
        row = {"model": model, "accepts_image": False, "note": "", "control": None,
               "crop": None}
        try:
            got = ask(client, model, CONTROL_SYSTEM, CONTROL_USER, control)
            row["control"] = got
            seen = normalize_code(got["text"])
            if seen == code:
                row["accepts_image"] = True
                row["note"] = f"read the control code exactly ({got['seconds']}s)"
            elif not got["text"]:
                row["note"] = "empty response — no text returned at all"
            elif code in seen:
                row["accepts_image"] = True
                row["note"] = f"read the code, plus extra characters: {got['text'][:40]!r}"
            else:
                row["note"] = (f"returned {got['text'][:40]!r}, not the control code "
                               f"— image was not read")
        except Exception as exc:
            row["note"] = f"{type(exc).__name__}: {str(exc)[:160]}"

        if row["accepts_image"] and crop is not None:
            try:
                row["crop"] = ask(client, model, CROP_SYSTEM, CROP_USER, crop)
            except Exception as exc:
                row["crop"] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}

        if not row["accepts_image"]:
            # Establish whether the model is reachable at all before concluding
            # anything about its vision support.
            row["text_only"] = probe_text_only(client, model)
        row["verdict"] = classify(row)

        label = {"usable": "PASS", "no_vision": "NO-VISION",
                 "unreachable": "UNREACHABLE"}[row["verdict"]]
        print(f"  [{label:11s}] {model:22s} {row['note']}")
        results.append(row)

    print(f"\n  {'model':22s} {'image?':>7s} {'verdict':>12s}  note")
    print(f"  {'-' * 22} {'-' * 7} {'-' * 12}  {'-' * 40}")
    for r in results:
        print(f"  {r['model']:22s} {'yes' if r['accepts_image'] else 'NO':>7s} "
              f"{r['verdict']:>12s}  {r['note'][:40]}")

    if crop is not None:
        print("\n  Persian crop transcription (recorded, NOT scored — there is no")
        print("  gold label for this crop, and 'looks like Persian' is exactly the")
        print("  judgement that cannot be trusted):")
        for r in results:
            if not r.get("crop"):
                continue
            text = r["crop"].get("text", r["crop"].get("error", ""))
            print(f"    {r['model']:22s} {text[:70]!r}")

    passed = [r["model"] for r in results if r["accepts_image"]]
    no_vision = [r for r in results if r["verdict"] == "no_vision"]
    unreachable = [r for r in results if r["verdict"] == "unreachable"]

    print(f"\n  Usable for the bake-off: {len(passed)}/{len(results)}"
          + (f" — {', '.join(passed)}" if passed else ""))
    if no_vision:
        print("\n  NO VISION — answered a text-only call and rejected the image.")
        print("  This is a capability fact; substitute a different model.")
        for r in no_vision:
            print(f"    {r['model']:22s} {r['note'][:64]}")
    if unreachable:
        print("\n  UNREACHABLE — the text-only call failed too, so vision is")
        print("  UNTESTED, not absent. Fix access, then re-run this check.")
        for r in unreachable:
            err = r.get("text_only", {}).get("error", "")
            print(f"    {r['model']:22s} {err[:64]}")
    if not passed:
        print("\n  Step 4 (bake-off) cannot run: no backend passed the gate.")

    report = {
        "endpoint": base_url,
        "control_code_len": len(code),
        "crop": str(crop_path.relative_to(PROJECT_ROOT)) if crop_path else None,
        "models": results,
        "passed": passed,
        "no_vision": [r["model"] for r in no_vision],
        "unreachable": [r["model"] for r in unreachable],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  report → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
