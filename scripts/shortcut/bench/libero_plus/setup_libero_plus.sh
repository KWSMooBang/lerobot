#!/usr/bin/env bash
# LIBERO-Plus client venv for sc-bench (T012). No sudo / apt: a missing system piece is printed.
#
#   scripts/shortcut/bench/libero_plus/setup_libero_plus.sh [--venv DIR] [--assets-zip PATH] [--skip-assets]
#                                                          [--skip-selftest]
#
# LIBERO-Plus installs as the `libero` package (it replaces vanilla LIBERO), so it lives in its own venv
# (default <repo>/.venvs/libero_plus), never in the main one. What it does:
#  1. uv venv (Python 3.12) + the fork's runtime deps, pinned to the main venv's simulator versions
#     (mujoco 3.8.1, robosuite 1.4.0, numpy 2.2.6: same physics / renderer as our C0 runs), CPU torch
#     (only torch.load of init states), websockets + msgpack for the policy-server client
#  2. clone sylvestf/LIBERO-plus at $LIBERO_PLUS_SHA into <venv>/src/LIBERO-plus, `pip install --no-deps -e`
#  3. assets.zip (Sylvest/LIBERO-plus, 6.4 GB) → <src>/libero/libero/assets (the fork resolves scene XMLs
#     package-relative, whatever config.yaml says)
#  4. LIBERO config at <venv>/libero_config/config.yaml (LIBERO_CONFIG_PATH; ~/.libero is left alone —
#     the main venv's hf-libero uses it)
#  5. checks ImageMagick for Wand (env_wrapper.py imports it at module level; only the Sensor Noise motion
#     blur, noise 1–10, calls it): apt libmagickwand-6.q16-6. Missing → printed, the worker stubs Wand and
#     refuses the motion-blur variants
#  6. <venv>/sc_env.json (paths the worker exports before `import libero`) + an import / env self-test
set -euo pipefail

LIBERO_PLUS_SHA=${LIBERO_PLUS_SHA:-4976dc30028e805ff8094b55501d532c48fec182}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
VENV=$REPO/.venvs/libero_plus
ZIP=""; ASSETS=1; SELFTEST=1
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV=$2; shift ;;
    --assets-zip) ZIP=$2; shift ;;
    --skip-assets) ASSETS=0 ;;
    --skip-selftest) SELFTEST=0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
VENV=$(realpath -m "$VENV")
SRC=$VENV/src/LIBERO-plus
PY=$VENV/bin/python
note() { echo "[libero_plus] $*"; }
command -v uv >/dev/null || { note "MISSING: uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }
command -v unzip >/dev/null || [ "$ASSETS" = 0 ] || { note "MISSING: unzip (apt: unzip)"; exit 1; }

# ------------------------------------------------------------ 1. venv + deps
[ -x "$PY" ] || uv venv -q --allow-existing "$VENV" --python 3.12
# --torch-backend cpu: overrides a global UV_TORCH_BACKEND=cu1xx (CUDA torch would add ~5 GB for nothing)
uv pip install -q --python "$PY" --torch-backend cpu torch \
  "numpy==2.2.6" "mujoco==3.8.1" "robosuite==1.4.0" "bddl==1.0.1" "easydict==1.13" "future>=0.18.2" \
  "matplotlib>=3.5.3" "opencv-python==4.13.0.92" "Wand==0.6.13" "scikit-image" "scipy" \
  "gym==0.26.2" "h5py" "termcolor" "cloudpickle" "pyyaml" "imageio" "imageio-ffmpeg" \
  "websockets>=13" "msgpack>=1.0"

# ------------------------------------------------------------ 2. LIBERO-Plus source
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone -q https://github.com/sylvestf/LIBERO-plus.git "$SRC"
fi
git -C "$SRC" fetch -q origin 2>/dev/null || true
git -C "$SRC" checkout -q "$LIBERO_PLUS_SHA"
uv pip install -q --python "$PY" --no-deps -e "$SRC"
LIB=$SRC/libero/libero

# ------------------------------------------------------------ 3. assets
if [ "$ASSETS" = 1 ] && [ ! -d "$LIB/assets/new_objects" ]; then
  DL=$VENV/dl
  mkdir -p "$DL"
  if [ -z "$ZIP" ]; then
    "$PY" -c "from huggingface_hub import hf_hub_download" 2>/dev/null || uv pip install -q --python "$PY" huggingface_hub
    ZIP=$("$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('Sylvest/LIBERO-plus', 'assets.zip', repo_type='dataset', local_dir='$DL'))")
  fi
  note "assets.zip: $(du -h "$ZIP" | cut -f1)"
  rm -rf "$DL/extract" && mkdir -p "$DL/extract"
  unzip -q "$ZIP" -d "$DL/extract"
  # the zip holds the tree under a deep author-internal prefix: take the first `assets` dir
  A=$(find "$DL/extract" -type d -name assets | head -1)
  [ -n "$A" ] || { note "no assets/ dir in $ZIP"; exit 1; }
  rm -rf "$LIB/assets"
  mv "$A" "$LIB/assets"
  rm -rf "$DL/extract"
  [ "$ZIP" = "$DL/assets.zip" ] && rm -rf "$DL"  # our own download (6 GB): no longer needed
fi
[ -d "$LIB/assets" ] && note "assets: $(du -sh "$LIB/assets" | cut -f1)"

# ------------------------------------------------------------ 4. LIBERO config (not ~/.libero)
CFG=$VENV/libero_config
mkdir -p "$CFG"
cat > "$CFG/config.yaml" <<EOF
assets: $LIB/assets
bddl_files: $LIB/bddl_files
benchmark_root: $LIB
datasets: $LIB/../datasets
init_states: $LIB/init_files
EOF

# ------------------------------------------------------------ 5. ImageMagick (Wand)
MAGICK=1
if ! ldconfig -p 2>/dev/null | grep -q "libMagickWand"; then
  MAGICK=0
  note "MISSING (not fatal): libMagickWand — ask the admin for apt package libmagickwand-6.q16-6."
  note "  Without it the worker stubs Wand: every variant runs except Sensor Noise 1–10 (motion blur)."
fi

# ------------------------------------------------------------ 6. sc_env.json + self-test
"$PY" - "$VENV/sc_env.json" <<EOF
import json, sys
json.dump({"libero_config_path": "$CFG", "magick": bool($MAGICK), "src": "$SRC",
           "sha": "$LIBERO_PLUS_SHA"}, open(sys.argv[1], "w"), indent=2)
EOF
note "venv: $(du -sh "$VENV" --exclude=src --exclude=dl | cut -f1) (+ src/assets $(du -sh "$SRC" | cut -f1))"
if [ "$SELFTEST" = 1 ]; then
  MUJOCO_GL=${MUJOCO_GL:-egl} "$PY" "$REPO/scripts/shortcut/bench/libero_plus/lp_env.py" selftest
fi
note "done: $PY"
