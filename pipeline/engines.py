"""Pluggable line-OCR engines.

Every engine takes one line crop (PIL RGB) and returns ``(text, confidence)``,
where confidence is in [0, 1] or ``None`` when the engine cannot report one.
Select with env ``OCR_ENGINE``:

- ``qwen_local`` — Qwen3-VL-2B Persian line-OCR (best measured: 62.3% char
  accuracy on the gold set), optional LoRA adapter via ``QWEN_ADAPTER_PATH``.
- ``vlm``        — OpenAI-compatible vision API (``pipeline/ocr.py``).
- ``fake``       — deterministic text, no model; for tests and UI demos.

Generation settings of ``qwen_local`` match ``scripts/eval_local_hf_gold.py``
exactly so the measured baseline applies. Change them only through the
evaluation gate (``.claude/skills/ocr-eval-gate``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from PIL import Image

QWEN_DEFAULT_MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
QWEN_PROMPT = "Transcribe the Persian text in this image. Output only the text."


class OCREngine(Protocol):
    version: str

    def recognize(self, image: Image.Image) -> tuple[str, float | None]: ...


class FakeEngine:
    """Returns a fixed sentence with a deliberate, repeatable OCR mistake.

    The mistake ("هسادجاری" for "حسابداری", taken from a real model error in
    REPORT_dataset_and_bakeoff.md §7) lets the demo and tests show the
    correction memory learning from a student edit.
    """

    version = "fake@1"

    def __init__(self, text: str | None = None) -> None:
        self.text = text or os.environ.get("FAKE_OCR_TEXT", "اصول هسادجاری و معادله اساسی")

    def recognize(self, image: Image.Image) -> tuple[str, float | None]:
        # Narrow crops get lower confidence so the UI's low-confidence path is exercised.
        confidence = 0.95 if image.width >= 200 else 0.45
        return self.text, confidence


class VLMApiEngine:
    def __init__(self) -> None:
        self.version = f"vlm-api:{os.environ.get('VLM_MODEL', 'deepseek-chat')}"

    def recognize(self, image: Image.Image) -> tuple[str, float | None]:
        from pipeline.ocr import _pil_to_base64, _prepare_ocr_region, extract_text_with_vlm

        return extract_text_with_vlm(_pil_to_base64(_prepare_ocr_region(image))), None


@dataclass
class QwenSettings:
    """Every knob that can change recognized text. Defaults = the measured
    baseline (62.3% on gold); change a default only after scripts/eval_gate.py
    shows a gain under the promotion rule."""

    prompt: str = QWEN_PROMPT
    max_new_tokens: int = 96
    # >0: cap new tokens at crop_width / px_per_token (+8), never above max_new_tokens.
    px_per_token: float = 0.0
    repetition_penalty: float = 1.0
    # Preprocessing, applied to the crop before the model sees it.
    min_height: int = 0          # upscale crops shorter than this (keeps aspect ratio)
    clahe: bool = False          # contrast-limited histogram equalization on luminance
    # Routing: crops narrower than this many px are not read; returned as "" for review.
    route_min_width: int = 0
    # Replace repetition loops (e.g. "۵۵۵۵۵…") with "" instead of showing them.
    drop_loops: bool = False

    @classmethod
    def from_env(cls) -> "QwenSettings":
        s = cls()
        env = os.environ.get
        if env("QWEN_PROMPT"):
            s.prompt = env("QWEN_PROMPT")
        s.max_new_tokens = int(env("QWEN_MAX_NEW_TOKENS", s.max_new_tokens))
        s.px_per_token = float(env("QWEN_PX_PER_TOKEN", s.px_per_token))
        s.repetition_penalty = float(env("QWEN_REPETITION_PENALTY", s.repetition_penalty))
        s.min_height = int(env("OCR_MIN_CROP_HEIGHT_UPSCALE", s.min_height))
        s.clahe = env("OCR_CLAHE", "1" if s.clahe else "0") == "1"
        s.route_min_width = int(env("OCR_ROUTE_MIN_WIDTH", s.route_min_width))
        s.drop_loops = env("OCR_DROP_LOOPS", "1" if s.drop_loops else "0") == "1"
        return s


# Safety cap, not a tuning knob: above ~2.4M px the vision tokens exceed the
# processor's max_length and generate() raises. Every legible gold crop is
# below 1.7M px, so the cap leaves the measured baseline untouched; it only
# shrinks page-sized blocks that segmentation mistook for a line.
# Height is capped too: a near-square page block at 1.8M px is ~7k vision
# patches under full attention and ran MPS out of memory (16 GB Mac). The
# tallest legible gold line is 750 px.
MAX_CROP_PIXELS = 1_800_000
MAX_CROP_HEIGHT = 800


def cap_pixels(image: Image.Image) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= MAX_CROP_PIXELS and image.height <= MAX_CROP_HEIGHT:
        return image
    scale = min((MAX_CROP_PIXELS / pixels) ** 0.5, MAX_CROP_HEIGHT / image.height)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)


def preprocess_crop(image: Image.Image, settings: QwenSettings) -> Image.Image:
    if settings.clahe:
        import cv2
        import numpy as np

        lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    if settings.min_height and image.height < settings.min_height:
        scale = settings.min_height / image.height
        image = image.resize((max(1, round(image.width * scale)), settings.min_height), Image.LANCZOS)
    return image


def is_repetition_loop(text: str) -> bool:
    """A decoding loop: one short unit (1–4 chars) repeated to fill most of the
    output, e.g. "٥٥٥٥٥٥…", "۲۶۲۶۲۶…", "1000000…". Real lines never look like this."""
    s = "".join(text.split())
    if len(s) < 12:
        return False
    for unit in range(1, 5):
        best = run = 1
        for i in range(unit, len(s)):
            run = run + 1 if s[i] == s[i - unit] else 1
            best = max(best, run)
        if best + unit >= 0.7 * len(s) and best >= 10:
            return True
    return False


class QwenLocalEngine:
    def __init__(self, settings: QwenSettings | None = None, adapter: str | None = None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = os.environ.get("QWEN_MODEL", QWEN_DEFAULT_MODEL)
        device = os.environ.get("QWEN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.settings = settings or QwenSettings.from_env()

        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=device)
        adapter = (adapter if adapter is not None else os.environ.get("QWEN_ADAPTER_PATH", "")).strip()
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model
        self._torch = torch
        slug = model_id.rstrip("/").split("/")[-1].lower()
        self.version = f"{slug}+{Path(adapter).name if adapter else 'base'}"

    def recognize(self, image: Image.Image) -> tuple[str, float | None]:
        torch = self._torch
        s = self.settings
        if s.route_min_width and image.width < s.route_min_width:
            return "", None  # routed to review: too short to read reliably
        max_new = s.max_new_tokens
        if s.px_per_token > 0:
            max_new = min(max_new, 8 + int(image.width / s.px_per_token))
        image = cap_pixels(preprocess_crop(image, s))
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": s.prompt}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        gen_kwargs = {}
        if s.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = s.repetition_penalty
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                **gen_kwargs,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out.sequences[0][prompt_len:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        confidence = None
        if out.scores:
            logprobs = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )[0]
            finite = [float(v) for v in logprobs if math.isfinite(float(v))]
            if finite:
                # Geometric-mean token probability: a single bad token pulls it down.
                confidence = math.exp(sum(finite) / len(finite))
        if self.model.device.type == "mps":
            # The MPS cache grows across calls (~14 GB after 160 gold lines) until
            # a large crop OOMs on a 16 GB Mac; releasing it per line costs ms.
            del out, inputs
            torch.mps.empty_cache()
        if s.drop_loops and is_repetition_loop(text):
            return "", confidence
        return text, confidence


@lru_cache(maxsize=1)
def get_engine(name: str | None = None) -> OCREngine:
    name = (name or os.environ.get("OCR_ENGINE", "fake")).lower()
    if name == "fake":
        return FakeEngine()
    if name == "vlm":
        return VLMApiEngine()
    if name in {"qwen", "qwen_local"}:
        return QwenLocalEngine()
    raise ValueError(f"Unknown OCR_ENGINE '{name}' (expected qwen_local | vlm | fake)")
