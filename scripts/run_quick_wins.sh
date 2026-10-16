#!/usr/bin/env bash
# Phase 5 quick wins: one variable at a time, each compared with base_v1 on gold.
# Reports: dataset/reports/qw_*.json. Resumable (re-run the script after an interrupt).
# Close LM Studio and other heavy apps first.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv-qwen/bin/python}
export HF_HUB_OFFLINE=1 QWEN_DEVICE=${QWEN_DEVICE:-mps}

run() {
  label=$1; shift
  echo "=== $label $*"
  "$PY" scripts/eval_gate.py --label "$label" --champion base_v1 "$@" 2>&1 \
    | grep -vE "Warning|Loading weights|warn" | tail -8
}

# 1. prompts (a handful only: gold is not a search space)
run qw_prompt_fa       --prompt "متن فارسی داخل این تصویر را بنویس. فقط متن را خروجی بده."
run qw_prompt_exact    --prompt "Read the single line of handwritten Persian text in this image and write it exactly as written. Output only the text."
run qw_prompt_short    --prompt "Transcribe the text in this image."
run qw_prompt_fa_exact --prompt "این تصویر یک خط از جزوهٔ دست‌نویس فارسی است. متن را دقیقاً همان‌طور که نوشته شده بنویس و چیزی اضافه نکن."
# 2. generation length / repetition
run qw_len_w15         --px-per-token 15
run qw_rep105          --repetition-penalty 1.05
run qw_rep110          --repetition-penalty 1.1
# 3-4. crop preprocessing
run qw_upscale64       --min-height 64
run qw_clahe           --clahe
# 5. routing / rejection (what is NOT read)
run qw_drop_loops      --drop-loops
run qw_route_w150      --route-min-width 150
echo "=== done"
