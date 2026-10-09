#!/usr/bin/env python3
"""Cache Qwen2.5-VL and Qwen3-VL predictions for sealed new_gold lines.

Stores rows in labeling/labels.db (model_preds) and JSONL under
``labeling/exports/preds_{model}.jsonl``.

Models:
  * qwen3-vl-2b-persian — local transformers (default bakeoff model)
  * qwen2.5-vl-7b — LM Studio OpenAI-compatible API on 127.0.0.1 (GGUF),
    optional HuggingFace if ``--hf-qwen25`` is set

Usage:
  python -m labeling.precompute_preds --models qwen3
  python -m labeling.precompute_preds --models qwen25,qwen3
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeling.db import connect, count_model_preds, upsert_model_pred  # noqa: E402
from labeling.image_ops import resize_for_vlm  # noqa: E402
from labeling.queue import load_manifest  # noqa: E402

QWEN3_ID = "qwen3-vl-2b-persian"
QWEN25_ID = "qwen2.5-vl-7b"
DEFAULT_QWEN3_HF = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
PROMPT = "Transcribe the Persian text in this image. Output only the text."
EXPORT_DIR = PROJECT_ROOT / "labeling" / "exports"


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _qwen3_infer(processor, model, img: Image.Image) -> str:
    import torch

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"truncation": False},
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return processor.decode(
        out[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()


def run_qwen3(
    manifest: list[dict],
    conn,
    *,
    device: str,
    limit: int = 0,
    max_long_side: int = 1024,
) -> int:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    model_id = os.environ.get("QWEN_OCR_MODEL", DEFAULT_QWEN3_HF)
    print(f"[preds] loading {model_id} on {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.float16, device_map=device
    )
    model.eval()

    export = EXPORT_DIR / f"preds_{QWEN3_ID}.jsonl"
    # Resume: skip line_ids already in DB
    existing = {
        r["line_id"]
        for r in conn.execute(
            "SELECT line_id FROM model_preds WHERE model_id=?", (QWEN3_ID,)
        )
    }
    n = 0
    rows = manifest[: limit or None]
    for i, row in enumerate(rows):
        lid = row["line_id"]
        if lid in existing:
            continue
        path = PROJECT_ROOT / (row.get("crop_path") or row["image_path"])
        img = resize_for_vlm(Image.open(path).convert("RGB"), max_long_side=max_long_side)
        try:
            text = _qwen3_infer(processor, model, img)
        except ValueError as exc:
            if "image` token count" in str(exc) and max_long_side > 512:
                img = resize_for_vlm(img, max_long_side=512)
                text = _qwen3_infer(processor, model, img)
            else:
                raise
        upsert_model_pred(conn, line_id=lid, model_id=QWEN3_ID, text=text)
        _append_jsonl(
            export,
            {"line_id": lid, "image_path": row["image_path"], "model_id": QWEN3_ID, "text": text},
        )
        n += 1
        if n % 10 == 0 or i + 1 == len(rows):
            print(f"[preds] qwen3 {i+1}/{len(rows)} new={n}")
    return n


def run_qwen25_lmstudio(
    manifest: list[dict],
    conn,
    *,
    base_url: str,
    model: str,
    limit: int = 0,
) -> int:
    export = EXPORT_DIR / f"preds_{QWEN25_ID}.jsonl"
    existing = {
        r["line_id"]
        for r in conn.execute(
            "SELECT line_id FROM model_preds WHERE model_id=?", (QWEN25_ID,)
        )
    }
    n = 0
    rows = manifest[: limit or None]
    for i, row in enumerate(rows):
        lid = row["line_id"]
        if lid in existing:
            continue
        path = PROJECT_ROOT / (row.get("crop_path") or row["image_path"])
        from io import BytesIO

        img = resize_for_vlm(Image.open(path).convert("RGB"), max_long_side=1024)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 128,
        }
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            print(f"[preds] qwen25 LM Studio unreachable: {exc}")
            print("  Start LM Studio with Qwen2.5-VL GGUF, or skip with --models qwen3")
            return n
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        upsert_model_pred(conn, line_id=lid, model_id=QWEN25_ID, text=text)
        _append_jsonl(
            export,
            {"line_id": lid, "image_path": row["image_path"], "model_id": QWEN25_ID, "text": text},
        )
        n += 1
        if n % 10 == 0 or i + 1 == len(rows):
            print(f"[preds] qwen25 {i+1}/{len(rows)} new={n}")
        time.sleep(0.05)
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="qwen3", help="Comma list: qwen3,qwen25")
    p.add_argument("--device", default=os.environ.get("QWEN_OCR_DEVICE", "mps"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lmstudio-base", default="http://127.0.0.1:1234/v1")
    p.add_argument("--lmstudio-model", default="qwen25vl")
    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "labeling" / "labels.db")
    p.add_argument(
        "--max-long-side",
        type=int,
        default=1024,
        help="Resize crops before VLM (avoids wide-line token truncation)",
    )
    args = p.parse_args(argv)

    manifest = load_manifest()
    print(f"[preds] new_gold lines={len(manifest)}")
    conn = connect(args.db)
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}

    if "qwen3" in wanted:
        run_qwen3(
            manifest,
            conn,
            device=args.device,
            limit=args.limit,
            max_long_side=args.max_long_side,
        )
        print(f"[preds] db count {QWEN3_ID}={count_model_preds(conn, QWEN3_ID)}")
    if "qwen25" in wanted:
        run_qwen25_lmstudio(
            manifest,
            conn,
            base_url=args.lmstudio_base,
            model=args.lmstudio_model,
            limit=args.limit,
        )
        print(f"[preds] db count {QWEN25_ID}={count_model_preds(conn, QWEN25_ID)}")

    # Mirror into manifest model_text for prefill convenience (qwen3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
