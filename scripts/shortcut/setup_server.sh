#!/usr/bin/env bash
# Prepare a new GPU server for scripts/shortcut (no sudo, no apt: missing system pieces are printed).
#
#   scripts/shortcut/setup_server.sh [--gpus 0,1,2,3] [--skip-download] [--skip-selftest] [--skip-train-data]
#
# 1. uv venv, Python 3.12 (draccus fails to parse configs on 3.14), extras libero + pi + smolvla + training,
#    local policy plugins (scripts/shortcut/policies/lerobot_policy_*) installed editable, no deps
# 2. LIBERO config + assets (lerobot/libero-assets), both checkpoints, PaliGemma tokenizer
#    (google/paligemma-3b-pt-224 is gated: falls back to the public big_vision sentencepiece model)
# 2b. training (scripts/shortcut/train, sc-train): HuggingFaceVLA/libero at the pinned revision (~35 GB, PNG
#    frames in parquet) -> $SC_OUT_ROOT/data/hf_libero, preset base weights (pi05_libero, pi0_base, SmolVLM2),
#    and an sc-train --dry-run per preset (lerobot parses + validates each config)
# 3. EGL check per GPU; common.py points glvnd at the NVIDIA EGL vendor only when glvnd lacks it
# 4. per-GPU self-test (in parallel): env reset + render in subprocesses, one batched π0.5 chunk, one step
set -euo pipefail

GPUS=all; DOWNLOAD=1; SELFTEST=1; TRAIN_DATA=1
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS=$2; shift ;;
    --skip-download) DOWNLOAD=0 ;;
    --skip-selftest) SELFTEST=0 ;;
    --skip-train-data) TRAIN_DATA=0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
LOG_DIR=${LOG_DIR:-./outputs/VLA-shortcut-exps/setup}
mkdir -p "$LOG_DIR"
fail=0
note() { echo "[setup] $*"; }
missing() { echo "[setup] MISSING: $*"; fail=1; }

# ------------------------------------------------------------ tools
if ! command -v uv >/dev/null; then
  missing "uv. User-level install (no sudo): curl -LsSf https://astral.sh/uv/install.sh | sh ; then re-run"
  exit 1
fi
command -v nvidia-smi >/dev/null || { missing "nvidia-smi (NVIDIA driver)"; exit 1; }
if [ "$GPUS" = all ]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
fi
note "GPUs: $GPUS"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/[setup]   /'
note "CPU cores: $(nproc)"

# EGL / GL user-space libraries (MuJoCo offscreen rendering)
for lib in libEGL.so.1 libEGL_nvidia.so.0 libGL.so.1; do
  if ! ldconfig -p 2>/dev/null | grep -q "$lib"; then
    case "$lib" in
      libEGL_nvidia.so.0) missing "$lib: comes with the NVIDIA driver user-space libraries (ask the admin)" ;;
      *) missing "$lib: ask the admin for apt packages libegl1 libgl1 libglvnd0 (not installed by this script)" ;;
    esac
  fi
done

# ------------------------------------------------------------ venv
note "uv sync (Python 3.12, extras libero + pi + smolvla + training)"
uv sync --locked --python 3.12 --extra libero --extra pi --extra smolvla --extra training
# `uv sync` removes packages not in the lockfile, so the plugins are (re)installed after it. sc-train itself
# imports them from source; the install is for lerobot's own plugin discovery (lerobot-eval, policy server).
for p in scripts/shortcut/policies/lerobot_policy_*/; do
  note "policy plugin $(basename "$p") (editable, no deps)"
  uv pip install --quiet --no-deps -e "$p"
done

# ------------------------------------------------------------ downloads
if [ "$DOWNLOAD" = 1 ]; then
  note "LIBERO config/assets, checkpoints, tokenizer"
  uv run python - <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, "scripts/shortcut")
from common import TOKENIZER_DIR, USER_TOKENIZER_DIR, ensure_libero_config
from huggingface_hub import snapshot_download

print("[setup] LIBERO config:", ensure_libero_config())
for repo in ("lerobot/pi05_libero_finetuned", "lerobot/pi05_base"):
    print("[setup] checkpoint", repo, "->", snapshot_download(repo))
if any((Path(d) / "tokenizer.model").exists() for d in (TOKENIZER_DIR, USER_TOKENIZER_DIR)):
    print("[setup] tokenizer: found")
else:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        print("[setup] tokenizer: google/paligemma-3b-pt-224 (HF access ok)")
    except Exception:
        import urllib.request

        USER_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
        urllib.request.urlretrieve(url, USER_TOKENIZER_DIR / "tokenizer.model")  # nosec B310
        print("[setup] tokenizer: gated on HF; downloaded", url, "->", USER_TOKENIZER_DIR)
EOF
fi

# ------------------------------------------------------------ training data + base weights (sc-train)
if [ "$DOWNLOAD" = 1 ] && [ "$TRAIN_DATA" = 1 ]; then
  OUT_ROOT=${SC_OUT_ROOT:-./outputs/VLA-shortcut-exps}
  note "training data -> $OUT_ROOT/data/hf_libero (~35 GB) and preset base weights"
  SC_OUT_ROOT=$OUT_ROOT uv run python - <<'EOF'
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "scripts/shortcut/train")
from huggingface_hub import snapshot_download
from sc_train import DATASET_REPO, DATASET_REV

dest = Path(os.environ["SC_OUT_ROOT"]) / "data" / "hf_libero"
free_gb = shutil.disk_usage(dest.parent if dest.parent.exists() else Path(".")).free / 2**30
if (dest / "meta" / "info.json").exists() and len(list(dest.glob("data/*/*.parquet"))) >= 377:
    print("[setup] training data: found", dest)
elif free_gb < 40:
    print(f"[setup] MISSING: training data needs ~35 GB, {free_gb:.0f} GB free at {dest.parent}")
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(DATASET_REPO, repo_type="dataset", revision=DATASET_REV, local_dir=dest)
    print("[setup] training data:", DATASET_REPO, DATASET_REV[:8], "->", dest)
for repo in ("lerobot/pi05_libero", "lerobot/pi0_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    print("[setup] base weights", repo, "->", snapshot_download(repo))
EOF
  for preset in scripts/shortcut/train/presets/*.yaml; do
    name=$(basename "$preset" .yaml)
    if out=$(SC_OUT_ROOT=$OUT_ROOT scripts/shortcut/train/sc-train --preset "$name" --run-id setup-check \
             --gpus "$GPUS" --allow-any-gpu --wandb off --dry-run 2>&1); then
      note "sc-train preset $name: config OK"
    else
      missing "sc-train preset $name does not validate:"; echo "$out" | tail -5 | sed 's/^/[setup]   /'
    fi
  done
fi

# ------------------------------------------------------------ EGL per GPU
IFS=, read -r -a G_A <<< "$GPUS"
for g in "${G_A[@]}"; do
  if out=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g \
           uv run python scripts/shortcut/parallel/selftest.py --egl-only 2>&1); then
    note "EGL GPU $g: $(echo "$out" | grep -o '"renderer": "[^"]*"')$(echo "$out" | grep -q '"egl_vendor_override": "/' && echo ' (NVIDIA vendor JSON override applied)')"
  else
    missing "EGL on GPU $g failed:"; echo "$out" | sed 's/^/[setup]   /'
  fi
done

# ------------------------------------------------------------ self-test
if [ "$SELFTEST" = 1 ] && [ "$fail" = 0 ]; then
  note "self-test on GPUs $GPUS (model load + 2 envs + 2 chunks each; logs in $LOG_DIR)"
  pids=()
  for g in "${G_A[@]}"; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g \
      uv run python scripts/shortcut/parallel/selftest.py > "$LOG_DIR/selftest_gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    g=${G_A[$i]}
    if wait "${pids[$i]}"; then
      note "GPU $g self-test OK: $(grep -o '"stats": "[^"]*"' "$LOG_DIR/selftest_gpu$g.log")"
    else
      missing "GPU $g self-test failed, see $LOG_DIR/selftest_gpu$g.log"
    fi
  done
fi

if [ "$fail" = 0 ]; then
  note "ready. Next: scripts/shortcut/submit_multi.sh --dry-run --gpus $GPUS ... (parallel/README.md)"
else
  note "not ready: fix the MISSING items above (this script does not use sudo/apt)"
  exit 1
fi
