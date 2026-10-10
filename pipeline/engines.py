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


class QwenLocalEngine:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = os.environ.get("QWEN_MODEL", QWEN_DEFAULT_MODEL)
        device = os.environ.get("QWEN_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "96"))

        self.processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, device_map=device)
        adapter = os.environ.get("QWEN_ADAPTER_PATH", "").strip()
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
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": QWEN_PROMPT}],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
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
