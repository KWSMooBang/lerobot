#!/usr/bin/env bash
# T009: re-render the eval init states of all 40 tasks (T008 renderer) with the fixed LIBERO env, GPU 0 under flock.
#
#   scripts/shortcut/analysis/env_fix_render.sh <cache dir> [SC_LIBERO_RAW=0|1]
#
# <cache dir> gets render/<suite>_t<id>.npz; the rest of the T008 cache is symlinked in, so
# train_data_viz.py --sections G --cache <cache dir> redraws the training-vs-eval comparison.
set -euo pipefail
CACHE=$1
export SC_LIBERO_RAW=${2:-0}
SRC=${T008_CACHE:-/outputs/VLA-shortcut-exps/data/hf_libero_viz_cache}
mkdir -p "$CACHE/render"
for f in episodes.parquet extract_meta.json features.npz first256.npz last256.npz lowdim.npz progress.npz rep; do
  ln -sfn "$SRC/$f" "$CACHE/$f"
done
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src:scripts/shortcut
LOCK=""
if [ -d /queue/locks ]; then
  LOCK=/queue/locks/$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0).lock
fi
for s in libero_spatial libero_object libero_goal libero_10; do
  if [ -n "$LOCK" ]; then
    flock "$LOCK" uv run python scripts/shortcut/analysis/train_data_render.py --suite "$s" --cache "$CACHE"
  else
    uv run python scripts/shortcut/analysis/train_data_render.py --suite "$s" --cache "$CACHE"
  fi
done
