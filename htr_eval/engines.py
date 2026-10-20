"""One interface for every line-level HTR engine: PIL line crop → {"text", "candidates"}.

All engines run locally. Add a new one by writing a class with ``name`` and
``recognize`` and registering it in ENGINES.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
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
                 html: bool = False, adapter: str | None = None, dtype=None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.name, self._max, self._html = name, max_new_tokens, html
        self.PROMPT = prompt or self.PROMPT
        self._dev = _device()
        dtype = dtype or (torch.float32 if self._dev == "cpu" else torch.float16)
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype).to(self._dev).eval()
        if adapter:  # LoRA adapter from htr_eval/lora_train.py
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, adapter).eval()

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
        for attempt in range(4):  # Ollama answers 500 when a runner restarts; give it a moment
            r = requests.post(f"{self._base}/api/chat", json=body, timeout=600)
            if r.status_code < 500:
                break
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        return {"text": r.json()["message"]["content"].strip(), "candidates": None}



def _env_key(name: str) -> str | None:
    """Optional API key from the environment or .env (never printed)."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return os.environ.get(name, "").strip() or None


class HostedOCR:
    """Hosted engines for PUBLIC images only (the DastKhat bake-off). Missing key -> SystemExit(skip)."""

    def __init__(self, name: str, key_env: str) -> None:
        self.name, self._key = name, _env_key(key_env)
        if not self._key:
            raise SystemExit(f"SKIP {name}: {key_env} is not set")

    @staticmethod
    def _png(line: Image.Image) -> bytes:
        from io import BytesIO
        buf = BytesIO()
        line.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()


class GeminiRec(HostedOCR):
    """Gemini via Google AI Studio (free tier: rate limited, so 429 is retried with a pause)."""

    def __init__(self) -> None:
        super().__init__("gemini", "GEMINI_API_KEY")
        self._model = _env_key("GEMINI_BAKEOFF_MODEL") or "gemini-2.5-flash"  # .env GEMINI_MODEL is a retired model

    def recognize(self, line: Image.Image) -> dict:
        import base64
        import time

        import requests
        body = {"contents": [{"parts": [{"text": OllamaVision.PROMPT},
                                        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(self._png(line)).decode()}}]}],
                "generationConfig": {"temperature": 0}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        for wait in (6, 30, 60, 90, 120):  # first wait paces calls under the free-tier per-minute limit
            time.sleep(wait)
            r = requests.post(url, json=body, headers={"x-goog-api-key": self._key}, timeout=120)
            if r.status_code not in (429, 500, 503):
                break
        r.raise_for_status()
        parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return {"text": "".join(p.get("text", "") for p in parts).strip(), "candidates": None}


class OcrSpaceRec(HostedOCR):
    """OCR.space Engine 3 (handwriting). Engine 3 auto-detects the language, so no `language` is sent.
    UNTESTED here: no OCR_SPACE_API_KEY on this machine."""

    def __init__(self) -> None:
        super().__init__("ocrspace", "OCR_SPACE_API_KEY")

    def recognize(self, line: Image.Image) -> dict:
        import requests
        r = requests.post("https://api.ocr.space/parse/image", headers={"apikey": self._key}, timeout=120,
                          data={"OCREngine": "3", "scale": "true"}, files={"file": ("line.png", self._png(line), "image/png")})
        r.raise_for_status()
        j = r.json()
        if j.get("IsErroredOnProcessing"):
            raise RuntimeError(f"OCR.space error: {str(j.get('ErrorMessage'))[:200]}")
        return {"text": " ".join(p.get("ParsedText", "") for p in j.get("ParsedResults", [])).strip(), "candidates": None}


class TesseractRec:
    """Control: Tesseract `fas` (printed-text OCR). traineddata lives in bakeoff/tessdata."""
    name = "tesseract-fas"

    def recognize(self, line: Image.Image) -> dict:
        import subprocess
        import tempfile
        root = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            line.convert("RGB").save(f.name)
            out = subprocess.run(["tesseract", f.name, "stdout", "-l", "fas", "--tessdata-dir", str(root / "bakeoff" / "tessdata"),
                                  "--psm", "7"], capture_output=True, text=True, check=True).stdout
        return {"text": " ".join(out.split()), "candidates": None}


class RapidOCRRec:
    """Control: RapidOCR with the Arabic-script PP-OCRv5 recognizer (covers Persian letters). Needs .venv-rapid."""
    name = "rapidocr-ar"

    def __init__(self) -> None:
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        self._ocr = RapidOCR(params={"Rec.lang_type": LangRec.ARABIC, "Rec.ocr_version": OCRVersion.PPOCRV5,
                                     "Rec.model_type": ModelType.MOBILE})

    def recognize(self, line: Image.Image) -> dict:
        import numpy as np
        res = self._ocr(np.array(line.convert("RGB")))
        if res.txts is None:
            return {"text": "", "candidates": None}
        parts = sorted(zip(res.boxes, res.txts), key=lambda p: -float(np.mean(np.array(p[0])[:, 0])))  # right to left
        return {"text": " ".join(t for _, t in parts).strip(), "candidates": None}

QWEN_FA = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"

ENGINES = {
    "tesseract-fas": TesseractRec,
    "rapidocr-ar": RapidOCRRec,
    "gemini": GeminiRec,
    "ocrspace": OcrSpaceRec,
    "paddleocr-vl": lambda: HFVisionLM("paddleocr-vl", "PaddlePaddle/PaddleOCR-VL", 128, "OCR:"),
    # the prompt is surya's training-time contract (surya/inference/prompts.py): do not reword it
    "surya2": lambda: HFVisionLM("surya2", "datalab-to/surya-ocr-2", 256, "OCR this block image to HTML.", html=True),
    "kraken-ppocrv6": KrakenRec,
    "surya": SuryaRec,
    "qwen3vl-lora": lambda: HFVisionLM("qwen3vl-lora", QWEN_FA, adapter=os.environ.get("LORA_ADAPTER", "htr_eval/lora/best")),
    "qwen3vl-fa": lambda: HFVisionLM("qwen3vl-fa", QWEN_FA),
}


def load_engine(name: str) -> Engine:
    if name.startswith("ollama-vision:"):
        model, _, prep = name[len("ollama-vision:"):].partition("@")
        return OllamaVision(name, model, tuple(prep.split(",")) if prep else ())
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return ENGINES[name]()
