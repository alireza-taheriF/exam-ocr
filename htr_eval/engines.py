"""One interface for every line-level HTR engine: PIL line crop → {"text", "candidates"}.

All engines run locally. Add a new one by writing a class with ``name`` and
``recognize`` and registering it in ENGINES.
"""
from __future__ import annotations

import os
from typing import Protocol

from PIL import Image


class Engine(Protocol):
    name: str

    def recognize(self, line: Image.Image) -> dict:  # {"text": str, "candidates": list[list[str]] | None}
        ...


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SuryaRec:
    """Baseline: Surya recognition (the project's pinned surya-ocr 0.6.x), Persian lang hint."""
    name = "surya"

    def __init__(self) -> None:
        from surya.model.recognition.model import load_model
        from surya.model.recognition.processor import load_processor
        from surya.ocr import run_recognition
        self._run, self._model, self._proc = run_recognition, load_model(), load_processor()

    def recognize(self, line: Image.Image) -> dict:
        w, h = line.size
        out = self._run([line], [["fa"]], self._model, self._proc, bboxes=[[[0, 0, w, h]]])
        return {"text": " ".join(t.text for t in out[0].text_lines), "candidates": None}


class HFVisionLM:
    """Any Hugging Face image-text-to-text OCR model (Qwen-VL fine-tunes etc.), greedy decoding."""
    PROMPT = "متن فارسی داخل این تصویر را عیناً رونویسی کن."

    def __init__(self, name: str, model_id: str, max_new_tokens: int = 96) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self._max = name, max_new_tokens
        self._dev = _device()
        dtype = torch.float32 if self._dev == "cpu" else torch.float16
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(self._dev).eval()

    def recognize(self, line: Image.Image) -> dict:
        import torch
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        prompt = self._proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self._proc(images=[line], text=[prompt], return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self._max, do_sample=False,
                                       repetition_penalty=1.05)
        text = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"text": text.strip(), "candidates": None}


ENGINES = {
    "surya": SuryaRec,
    "qwen3vl-fa": lambda: HFVisionLM("qwen3vl-fa", "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"),
}


def load_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return ENGINES[name]()
