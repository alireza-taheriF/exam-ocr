#!/usr/bin/env python3
"""LoRA fine-tune of the local Persian Qwen3-VL-2B OCR model on line crops.

Design choices (each answers a failure mode found in the earlier lora_v3 run, see
progress/STATUS.md "LoRA failure diagnosis"):

* Training and inference see the SAME pixels: no extra down-scaling (v3 shrank
  crops to <=512px at training time but inference used full size).
* Same chat template and prompt as scripts/run_ocr_preds.py. The prompt tokens
  are masked (-100); loss is computed on the answer tokens + end-of-turn only.
* LoRA only on the language model's attention projections; vision tower frozen.
* Replay: every batch is drawn from several named sources with fixed ratios
  (default: at least 30% general handwriting/printed replay).
* Periodic dev evaluation (greedy decode, CER on a fixed dev set); the best
  adapter is kept, not the last.
* `--sanity` runs without any training: (1) tokenization/label-mask checks,
  (2) step-zero equivalence — the PEFT-wrapped model with freshly initialised
  adapters must generate EXACTLY what the base model generates.

Leakage guard: every training row's image_path and source_document must be
absent from val_v3 / test_v3 / gold (and gold_v1.1 / dev files).

    HF_HUB_OFFLINE=1 .venv-qwen/bin/python training/finetune_qwen3vl_lora.py --sanity
    HF_HUB_OFFLINE=1 .venv-qwen/bin/python training/finetune_qwen3vl_lora.py \
        --mix claude=dataset/train_claude.jsonl:0.5 --mix hw=dataset/external/hw_lines/train_4k.jsonl:0.5 \
        --out models/lora_x --max-steps 600 --eval-every 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = "mohajesmaeili/Qwen3-VL-2B-Persian-Arabic-Ocr-v1.0"
PROMPT = "Transcribe the Persian text in this image. Output only the text."
DS = ROOT / "dataset"


def load_jsonl(p: Path) -> list[dict]:
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


# --------------------------------------------------------------------- data
def forbidden_sets() -> tuple[set[str], set[str]]:
    paths, docs = set(), set()
    for name in ("val_v3", "test_v3", "gold", "gold_v1.1", "dev_claude", "dev2_claude", "dev_todo", "dev2_todo"):
        p = DS / f"{name}.jsonl"
        if p.exists():
            for r in load_jsonl(p):
                paths.add(r["image_path"])
                if r.get("source_document") and name in ("val_v3", "test_v3", "gold", "gold_v1.1"):
                    docs.add(r["source_document"])
    return paths, docs


def load_source(path: Path, forbidden_paths, forbidden_docs, min_width=0, only_high=False, external=False):
    rows = []
    for r in load_jsonl(path):
        t = (r.get("text") or "").strip()
        if not t or r.get("illegible"):
            continue
        if only_high and r.get("confidence") not in (None, "high"):
            continue
        if not external:  # corpus rows must be provably outside every held-out set
            if r["image_path"] in forbidden_paths or r.get("source_document") in forbidden_docs:
                raise SystemExit(f"LEAKAGE: {r['image_path']} ({r.get('source_document')}) is in a held-out set")
        elif r["image_path"] in forbidden_paths:
            raise SystemExit(f"LEAKAGE: {r['image_path']}")
        rows.append({"image_path": r["image_path"], "text": t})
    return rows


class Batcher:
    """Draws examples from named sources with fixed probabilities."""

    def __init__(self, sources: dict[str, list[dict]], ratios: dict[str, float], seed: int):
        self.sources, self.rng = sources, random.Random(seed)
        self.names = list(sources)
        self.w = [ratios[n] for n in self.names]
        self.order = {n: [] for n in self.names}

    def _next(self, name):
        if not self.order[name]:
            self.order[name] = list(range(len(self.sources[name])))
            self.rng.shuffle(self.order[name])
        return self.sources[name][self.order[name].pop()]

    def draw(self, k):
        return [self._next(self.rng.choices(self.names, self.w)[0]) for _ in range(k)]


# ------------------------------------------------------------ tokenisation
def build_example(processor, image, text, prompt=PROMPT):
    """-> dict of tensors with `labels` masked on the prompt (answer + EOT kept)."""
    user = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]
    p = processor.apply_chat_template(user, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    f = processor.apply_chat_template(full, add_generation_prompt=False, tokenize=True,
                                      return_dict=True, return_tensors="pt")
    n_prompt = p["input_ids"].shape[1]
    assert (f["input_ids"][0, :n_prompt] == p["input_ids"][0]).all(), "prompt is not a prefix of full sequence"
    labels = f["input_ids"].clone()
    labels[:, :n_prompt] = -100
    f["labels"] = labels
    return f


def collate(items, pad_id):
    import torch
    out = {}
    L = max(i["input_ids"].shape[1] for i in items)
    for key, pad in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
        out[key] = torch.stack([torch.nn.functional.pad(i[key][0], (0, L - i[key].shape[1]), value=pad)
                                for i in items])
    # vision tensors concatenate along their leading (patch) axis
    for key in items[0]:
        if key in out or key in ("input_ids", "attention_mask", "labels"):
            continue
        if key == "pixel_values" or key == "image_grid_thw":
            out[key] = torch.cat([i[key] for i in items], 0)
        elif items[0][key].dim() == 2 and items[0][key].shape[1] == items[0]["input_ids"].shape[1]:
            out[key] = torch.stack([torch.nn.functional.pad(i[key][0], (0, L - i[key].shape[1]), value=0)
                                    for i in items])
    return out


# --------------------------------------------------------------- evaluation
def generate(model, processor, image, prompt=PROMPT, max_new=96, rep=1.0):
    import torch
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False, repetition_penalty=rep)
    return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def eval_dev(model, processor, dev_rows, max_new=96, rep=1.0, limit=0):
    from PIL import Image
    from training.eval.metrics import corpus_cer
    rows = dev_rows[:limit] if limit else dev_rows
    model.eval()
    hyps = [generate(model, processor, Image.open(ROOT / r["image_path"]).convert("RGB"), max_new=max_new, rep=rep)
            for r in rows]
    refs = [r["text"] for r in rows]
    return corpus_cer(refs, hyps), corpus_cer(refs, hyps, lenient=True), hyps


def load_model(device, dtype_name):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=dtype, device_map=device)
    return model, processor


LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


def wrap_lora(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_REGEX,
                     task_type="CAUSAL_LM", bias="none")
    return get_peft_model(model, cfg)


# ------------------------------------------------------------------ sanity
def sanity(args):
    from PIL import Image
    dev = load_source(DS / args.dev, set(), set(), external=True)[: args.sanity_n]
    model, processor = load_model(args.device, args.dtype)
    pad_id = processor.tokenizer.pad_token_id
    # 1) tokenisation / label masking
    ex = build_example(processor, Image.open(ROOT / dev[0]["image_path"]).convert("RGB"), dev[0]["text"])
    lab = ex["labels"][0]
    ans_ids = lab[lab != -100]
    decoded = processor.tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
    print("answer decoded from labels:", repr(decoded), "| expected:", repr(dev[0]["text"]))
    assert decoded == dev[0]["text"], "label tokens do not decode to the target text"
    print("last label token:", repr(processor.tokenizer.decode(ans_ids[-1:])), "(should be end-of-turn)")
    b = collate([ex, build_example(processor, Image.open(ROOT / dev[1]["image_path"]).convert("RGB"), dev[1]["text"])],
                pad_id)
    print("collate keys/shapes:", {k: tuple(v.shape) for k, v in b.items()})
    # 2) step-zero equivalence: base vs freshly wrapped (B=0) model
    imgs = [Image.open(ROOT / r["image_path"]).convert("RGB") for r in dev]
    base_out = [generate(model, processor, im) for im in imgs]
    peft_model = wrap_lora(model, args.rank, args.alpha, 0.0)
    peft_model.eval()
    lora_out = [generate(peft_model, processor, im) for im in imgs]
    same = sum(a == b for a, b in zip(base_out, lora_out))
    print(f"step-zero equivalence: {same}/{len(imgs)} identical generations")
    for a, b2 in zip(base_out, lora_out):
        if a != b2:
            print("  base:", repr(a), "\n  lora:", repr(b2))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print("trainable params:", trainable)
    return 0 if same == len(imgs) else 1


# ------------------------------------------------------------------ train
def train(args):
    import torch
    from PIL import Image
    fp, fd = forbidden_sets()
    sources, ratios = {}, {}
    for spec in args.mix:
        name, rest = spec.split("=", 1)
        path, ratio = rest.rsplit(":", 1)
        external = name.startswith("hw") or name.startswith("replay")
        sources[name] = load_source(ROOT / path, fp, fd, min_width=args.min_width,
                                    only_high=args.only_high, external=external)
        ratios[name] = float(ratio)
        print(f"source {name}: {len(sources[name])} rows, ratio {ratio}")
    assert abs(sum(ratios.values()) - 1) < 1e-6, "ratios must sum to 1"
    dev = load_source(DS / args.dev, set(), set(), external=True)
    model, processor = load_model(args.device, args.dtype)
    pad_id = processor.tokenizer.pad_token_id
    model = wrap_lora(model, args.rank, args.alpha, args.dropout)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapter weights
    model.print_trainable_parameters()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * max(0.0, 1 - s / max(1, args.max_steps)))
    batcher = Batcher(sources, ratios, args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cer0, lcer0, _ = eval_dev(model, processor, dev, rep=args.rep, limit=args.eval_limit)
    print(f"step 0  dev CER {cer0:.4f} lenient {lcer0:.4f}", flush=True)
    log = [{"step": 0, "dev_cer": cer0, "dev_cer_lenient": lcer0}]
    best = (cer0, 0)
    t0 = time.time()
    model.train()
    for step in range(1, args.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.grad_accum):
            items = [build_example(processor, Image.open(ROOT / r["image_path"]).convert("RGB"), r["text"])
                     for r in batcher.draw(args.batch)]
            batch = {k: v.to(model.device) for k, v in collate(items, pad_id).items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            print(f"step {step}: non-finite grad norm — skipping update", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        sched.step()
        if args.device == "mps":
            torch.mps.empty_cache()
        if step % 10 == 0:
            print(f"step {step} loss {total:.4f} gn {float(gn):.2f} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.max_steps:
            c, lc, _ = eval_dev(model, processor, dev, rep=args.rep, limit=args.eval_limit)
            model.train()
            log.append({"step": step, "loss": total, "dev_cer": c, "dev_cer_lenient": lc})
            print(f"step {step}  dev CER {c:.4f} lenient {lc:.4f}  (best {best[0]:.4f}@{best[1]})", flush=True)
            if c < best[0]:
                best = (c, step)
                model.save_pretrained(out / "best")
    (out / "train_log.json").write_text(json.dumps(
        {"args": vars(args), "log": log, "best_dev_cer": best[0], "best_step": best[1]}, indent=1))
    print("best", best)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--sanity-n", type=int, default=8)
    ap.add_argument("--mix", action="append", default=[], help="name=path.jsonl:ratio (name hw*/replay* = external data)")
    ap.add_argument("--dev", default="dev2_claude.jsonl")
    ap.add_argument("--out", default="models/lora_new")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--rep", type=float, default=1.0)
    ap.add_argument("--min-width", type=int, default=0)
    ap.add_argument("--only-high", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return sanity(args) if args.sanity else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
