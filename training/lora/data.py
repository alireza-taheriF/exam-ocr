"""Training data for the Qwen3-VL line-OCR LoRA.

Three kinds of source, all reduced to rows ``{"image_path", "text", "source"}``
with the image saved as a file (so one Dataset class serves every source):

- **local manifests** (``image_path`` + ``text`` JSONL): exported student
  corrections (``scripts/export_corrections.py`` → ``train.jsonl``), labeled
  val lines, etc. ``image_path`` may be relative to the project root or to the
  manifest's own directory. Sealed crops (gold, test split, new_gold) are
  refused — see ``training/sealed.py``.
- **MR3z4/persian-handwriting-ocr** — real Persian handwriting. Its HF rows are
  word boxes on page files with no Image column, so ``hf_rows`` cannot read it;
  build line crops on the Mac with ``scripts/build_hw_lines.py`` and pass the
  manifests as local data.
- **mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Small** — the base model's
  own training distribution; used as replay so the adapter does not forget it.
  Columns ``Image``/``Text``, splits ``Train``/``Test`` (checked 2026-10-13).

``hf_rows`` matches column and split names case-insensitively and prints the
features it found, so a wrong guess is visible in the Colab log; every choice
can be overridden on the command line.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HANDWRITING_DS = "MR3z4/persian-handwriting-ocr"
REPLAY_DS = "mohajesmaeili/Persian_Arabic_TextLine_Image_Ocr_Small"
TEXT_COLUMNS = ("text", "label", "transcription", "ground_truth", "gt", "sentence", "ocr_text", "line_text")
BBOX_COLUMNS = ("bbox", "box", "bounding_box", "coordinates")


class SealedCropError(ValueError):
    pass


def load_local(manifest: Path, sealed: set[str], source: str | None = None) -> list[dict]:
    rows = []
    for line in manifest.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        text = (r.get("text") or "").strip()
        if not text:
            continue  # illegible / unlabeled: nothing to learn from
        path = Path(r["image_path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path if (PROJECT_ROOT / path).exists() else manifest.parent / path
        if path.name in sealed:
            raise SealedCropError(f"{manifest}: {path.name} is a sealed evaluation crop; it must not be trained on")
        rows.append({"image_path": str(path), "text": text, "source": source or manifest.stem})
    return rows


def _pick_columns(features: dict, image_col: str | None, text_col: str | None, bbox_col: str | None):
    from datasets import Image as HFImage

    by_lower = {k.lower(): k for k in features}
    if image_col is None:
        image_col = next((k for k, v in features.items() if isinstance(v, HFImage)), None)
    if text_col is None:
        text_col = next((by_lower[k] for k in TEXT_COLUMNS if k in by_lower), None)
    if bbox_col is None:
        bbox_col = next((by_lower[k] for k in BBOX_COLUMNS if k in by_lower), None)
    if image_col is None or text_col is None:
        raise SystemExit(f"Cannot tell image/text columns from features {list(features)}; "
                         f"pass --*-image-col / --*-text-col")
    return image_col, text_col, bbox_col


def resolve_split(split: str, available: Iterable[str]) -> str:
    """``train`` → ``Train`` when the dataset capitalizes its split names."""
    available = list(available)
    if split in available:
        return split
    match = next((a for a in available if a.lower() == split.lower()), None)
    if match is None:
        raise SystemExit(f"split {split!r} not in {available}")
    return match


def _crop(image, bbox):
    """bbox as [x1, y1, x2, y2] or [x, y, w, h] (w/h detected when x2 < x1)."""
    x1, y1, a, b = [float(v) for v in bbox[:4]]
    x2, y2 = (a, b) if a > x1 and b > y1 else (x1 + a, y1 + b)
    return image.crop((int(x1), int(y1), int(x2), int(y2)))


def hf_rows(name: str, n: int, out_dir: Path, *, split: str = "train", seed: int = 0,
            image_col: str | None = None, text_col: str | None = None, bbox_col: str | None = None,
            source: str | None = None) -> list[dict]:
    """Stream ``n`` labeled lines from an HF dataset, saving each crop as a JPEG."""
    if n <= 0:
        return []
    from datasets import get_dataset_split_names, load_dataset

    split = resolve_split(split, get_dataset_split_names(name))
    ds = load_dataset(name, split=split, streaming=True)
    features = ds.features or {}
    image_col, text_col, bbox_col = _pick_columns(features, image_col, text_col, bbox_col)
    print(f"[data] {name}: features={list(features)} -> image={image_col} text={text_col} bbox={bbox_col}")
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        text = str(ex.get(text_col) or "").strip()
        image = ex.get(image_col)
        if not text or image is None:
            continue
        image = image.convert("RGB")
        if bbox_col and ex.get(bbox_col) is not None:
            image = _crop(image, ex[bbox_col])
        if image.width < 8 or image.height < 8:
            continue
        path = out_dir / f"{i:07d}.jpg"
        image.save(path, quality=92)
        rows.append({"image_path": str(path), "text": text, "source": source or name})
        if len(rows) >= n:
            break
    print(f"[data] {name}: kept {len(rows)}")
    return rows


def mix(in_domain: list[dict], handwriting: list[dict], replay: list[dict], *,
        replay_ratio: float, seed: int = 0) -> list[dict]:
    """In-domain + handwriting rows, plus replay rows = replay_ratio × that count."""
    rnd = random.Random(seed)
    core = in_domain + handwriting
    k = min(len(replay), int(round(replay_ratio * len(core)))) if core else len(replay)
    rows = core + rnd.sample(replay, k)
    rnd.shuffle(rows)
    return rows


def split_val(rows: list[dict], fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    n_val = int(len(rows) * fraction)
    val = {i for i in idx[:n_val]}
    return [r for i, r in enumerate(rows) if i not in val], [r for i, r in enumerate(rows) if i in val]


def manifest_hash(rows: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["image_path"]):
        h.update(Path(r["image_path"]).name.encode())
        h.update(b"\t")
        h.update(r["text"].encode())
        h.update(b"\n")
    return h.hexdigest()
