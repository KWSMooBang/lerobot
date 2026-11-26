#!/usr/bin/env bash
# T034 run 3: VLM collection (prefix-only, emb + pg00–pg17, frame stride 2, same 64 parts as the T031 expert acts).
# Waits for the expert sweep to finish first (one GPU job at a time). Re-running resumes (finished parts are kept).
set -uo pipefail
cd "$(dirname "$0")"
LOG=/outputs/VLA-shortcut-exps/sae/crosscoder/logs
until grep -q "sweep done\|smoke failed" $LOG/expert_sweep.log 2>/dev/null; do sleep 30; done
export SC_LIBERO_DATA_ROOT=/data0/woosung/datasets/VLA-shortcut-exps/libero/hf_libero
export HF_DATASETS_CACHE=/data0/woosung/datasets/VLA-shortcut-exps/libero/hf_cache
export HF_HUB_OFFLINE=1
T034_NOLOCK=1 ./run_gpu.sh collect_vlm ../../../.venv312/bin/python collect.py --model pi05lib --layers emb,pg:0-17 \
  --frame-stride 2 --acts-dir /data0/woosung/datasets/VLA-shortcut-exps/sae/vlm_acts/pi05lib_s2 --gpus 0 \
  --batch 32 --workers 6
echo "collect exit $?"
