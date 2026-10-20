"""One interface for every line-level HTR engine: PIL line crop → {"text", "candidates"}.

All engines run locally. Add a new one by writing a class with ``name`` and
``recognize`` and registering it in ENGINES.
"""
from __future__ import annotations

import os
import re
from html import unescape
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

    def __init__(self, name: str, model_id: str, max_new_tokens: int = 96, prompt: str | None = None,
                 html: bool = False) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self._max, self._html = name, max_new_tokens, html
        self.PROMPT = prompt or self.PROMPT
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
        if self._html:  # surya-ocr-2 answers in HTML
            text = " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())
        return {"text": text.strip(), "candidates": None}


class KrakenRec:
    """Kraken PP-OCRv6 (CTC line recognizer, CPU). Needs `.venv-kraken` (kraken>=7.1, python<3.14)."""
    name = "kraken-ppocrv6"
    MODEL = "htr_eval/models/kraken-ppocrv6-medium/medium.safetensors"

    def __init__(self) -> None:
        from pathlib import Path
        from kraken.configs import RecognitionInferenceConfig
        from kraken.tasks import RecognitionTaskModel
        self._cfg = RecognitionInferenceConfig(num_line_workers=0)
        self._model = RecognitionTaskModel.load_model(Path(__file__).resolve().parents[1] / self.MODEL)

    def recognize(self, line: Image.Image) -> dict:
        from kraken.containers import BBoxLine, Segmentation
        w, h = line.size
        seg = Segmentation(type="bbox", imagename=None, text_direction="horizontal-rl", script_detection=False,
                           lines=[BBoxLine(id="l0", bbox=(0, 0, w, h), text_direction="horizontal-rl")])
        rec = next(iter(self._model.predict(line, seg, self._cfg)), None)
        return {"text": rec.prediction.strip() if rec else "", "candidates": None}


class OllamaVision:
    """Any vision model served by local Ollama. Name: ``ollama-vision:<model>[@pad,up,gray]``."""
    PROMPT = ("متن دست‌نویس فارسی داخل این تصویر را دقیقاً همان‌طور که نوشته شده رونویسی کن؛ املا و ارقام نویسنده را حفظ کن. "
              "فقط خود متن را بنویس. اگر خوانا نیست رشتهٔ خالی برگردان.")

    def __init__(self, name: str, model: str, prep: tuple[str, ...] = ()) -> None:
        from urllib.parse import urlparse
        self.name, self._model, self._prep = name, model, prep
        self._base = os.environ.get("CORRECTOR_BASE_URL", "http://localhost:11434").rstrip("/")
        if urlparse(self._base).hostname not in {"localhost", "127.0.0.1"}:
            raise SystemExit(f"refusing non-local Ollama host {self._base!r}: nothing may leave this machine")

    def _preprocess(self, im: Image.Image) -> Image.Image:
        from PIL import ImageOps
        if "gray" in self._prep:
            im = ImageOps.autocontrast(im.convert("L"), cutoff=1).convert("RGB")
        if "up" in self._prep and im.height < 96:
            im = im.resize((round(im.width * 96 / im.height), 96), Image.Resampling.LANCZOS)
        if "pad" in self._prep:
            im = ImageOps.expand(im, 8, fill="white")
        return im

    def recognize(self, line: Image.Image) -> dict:
        import base64
        from io import BytesIO

        import requests
        buf = BytesIO()
        self._preprocess(line).save(buf, format="PNG")
        body = {"model": self._model, "stream": False, "options": {"temperature": 0, "num_predict": 160},
                "messages": [{"role": "user", "content": self.PROMPT,
                              "images": [base64.b64encode(buf.getvalue()).decode()]}]}
        if self._model.startswith("qwen3"):
            body["think"] = False
        r = requests.post(f"{self._base}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        return {"text": r.json()["message"]["content"].strip(), "candidates": None}


ENGINES = {
    "paddleocr-vl": lambda: HFVisionLM("paddleocr-vl", "PaddlePaddle/PaddleOCR-VL", 128, "OCR:"),
    # the prompt is surya's training-time contract (surya/inference/prompts.py): do not reword it
    "surya2": lambda: HFVisionLM("surya2", "datalab-to/surya-ocr-2", 256, "OCR this block image to HTML.", html=True),
    "kraken-ppocrv6": KrakenRec,
    "surya": SuryaRec,
    "qwen3vl-fa": lambda: HFVisionLM("qwen3vl-fa", "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"),
}


def load_engine(name: str) -> Engine:
    if name.startswith("ollama-vision:"):
        model, _, prep = name[len("ollama-vision:"):].partition("@")
        return OllamaVision(name, model, tuple(prep.split(",")) if prep else ())
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return ENGINES[name]()
