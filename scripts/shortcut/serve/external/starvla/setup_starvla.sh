#!/usr/bin/env bash
# StarVLA venv + pinned sources + checkpoint for the sc-policy/1 StarVLA server (T016). No sudo, no apt.
#
#   scripts/shortcut/serve/external/starvla/setup_starvla.sh [--root DIR] [--cpu] [--skip-download]
#
# DIR (default $STARVLA_ROOT, else $SC_OUT_ROOT/starvla, else /outputs/VLA-shortcut-exps/starvla):
#   DIR/src/starVLA          git clone of starVLA/starVLA at STARVLA_SHA (branch starVLA_dev)
#   DIR/venv                 uv venv (Python 3.10), torch 2.6.0 (cu124 wheels; --cpu: CPU wheels), StarVLA
#                            requirements needed for inference, msgpack + websockets. Separate from lerobot's venv.
#                            No flash-attn: StarVLA's Qwen3 wrapper forces attn_implementation="sdpa" at this sha.
#   DIR/models/Qwen3-VL-PI-LIBERO-4in1   HF StarVLA/Qwen3-VL-PI-LIBERO-4in1 @ CKPT_SHA (16.4 GB)
#   DIR/models/Qwen3-VL-4B-Instruct      HF Qwen/Qwen3-VL-4B-Instruct @ BASE_VLM_SHA (8.9 GB)
#   DIR/env.sh               STARVLA_ROOT / STARVLA_PYTHON / STARVLA_SRC / STARVLA_CKPT / STARVLA_BASE_VLM + shas
# Size: venv ≈ 9 GB (GPU wheels) + models ≈ 25 GB.
# The pins below are the single source of truth; server.py records them in hello / run identities.
set -euo pipefail

STARVLA_REPO=https://github.com/starVLA/starVLA.git
STARVLA_SHA=13d8ad43eb4128554c57227507055ecf9c3a62a2 # starVLA_dev 2026-10-31 (has the use_canonical_forward alias)
CKPT_REPO=StarVLA/Qwen3-VL-PI-LIBERO-4in1
CKPT_SHA=a461ce262ae7c8890d948e122c92fd06cd4803c7
BASE_VLM_REPO=Qwen/Qwen3-VL-4B-Instruct
BASE_VLM_SHA=ebb281ec70b05090aa6165b016eac8ec08e71b17

ROOT=${STARVLA_ROOT:-${SC_OUT_ROOT:-/outputs/VLA-shortcut-exps}/starvla}
CPU=0; DOWNLOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT=$2; shift ;;
    --cpu) CPU=1 ;;
    --skip-download) DOWNLOAD=0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
note() { echo "[setup-starvla] $*"; }
command -v uv >/dev/null || { echo "uv missing: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
mkdir -p "$ROOT/src" "$ROOT/models"
ROOT=$(cd "$ROOT" && pwd)

# ------------------------------------------------------------ sources
SRC=$ROOT/src/starVLA
if [ ! -d "$SRC/.git" ]; then
  note "clone $STARVLA_REPO -> $SRC"
  git clone -q -b starVLA_dev "$STARVLA_REPO" "$SRC"
fi
git -C "$SRC" fetch -q origin starVLA_dev || true
git -C "$SRC" checkout -q "$STARVLA_SHA"
[ "$(git -C "$SRC" rev-parse HEAD)" = "$STARVLA_SHA" ] || { echo "StarVLA checkout failed" >&2; exit 1; }
note "StarVLA at $STARVLA_SHA"

# ------------------------------------------------------------ venv
VENV=$ROOT/venv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.10 "$VENV"
PY=$VENV/bin/python
if [ "$CPU" = 1 ]; then
  TORCH_INDEX=https://download.pytorch.org/whl/cpu
else
  TORCH_INDEX=https://download.pytorch.org/whl/cu124
fi
note "torch 2.6.0 / torchvision 0.21.0 from $TORCH_INDEX"
uv pip install -q --python "$PY" --index-url "$TORCH_INDEX" torch==2.6.0 torchvision==0.21.0
note "StarVLA inference requirements ($HERE/requirements-starvla.txt)"
uv pip install -q --python "$PY" -r "$HERE/requirements-starvla.txt"
uv pip install -q --python "$PY" --no-deps -e "$SRC"

# ------------------------------------------------------------ models
CKPT_DIR=$ROOT/models/Qwen3-VL-PI-LIBERO-4in1
VLM_DIR=$ROOT/models/Qwen3-VL-4B-Instruct
if [ "$DOWNLOAD" = 1 ]; then
  note "download $CKPT_REPO@$CKPT_SHA and $BASE_VLM_REPO@$BASE_VLM_SHA"
  "$PY" - "$CKPT_REPO" "$CKPT_SHA" "$CKPT_DIR" "$BASE_VLM_REPO" "$BASE_VLM_SHA" "$VLM_DIR" <<'EOF'
import sys

from huggingface_hub import snapshot_download

a = sys.argv[1:]
for repo, rev, out in ((a[0], a[1], a[2]), (a[3], a[4], a[5])):
    snapshot_download(repo, revision=rev, local_dir=out)
    print(f"[setup-starvla] {repo}@{rev} -> {out}")
EOF
  printf '%s\n' "$CKPT_SHA" >"$CKPT_DIR/.sc_revision"
  printf '%s\n' "$BASE_VLM_SHA" >"$VLM_DIR/.sc_revision"
fi

cat >"$ROOT/env.sh" <<EOF
# written by setup_starvla.sh $(date -Iseconds)
export STARVLA_ROOT=$ROOT
export STARVLA_PYTHON=$PY
export STARVLA_SRC=$SRC
export STARVLA_SHA=$STARVLA_SHA
export STARVLA_CKPT=$CKPT_DIR/checkpoints/steps_100000_pytorch_model.pt
export STARVLA_CKPT_SHA=$CKPT_SHA
export STARVLA_BASE_VLM=$VLM_DIR
export STARVLA_BASE_VLM_SHA=$BASE_VLM_SHA
EOF

# ------------------------------------------------------------ import check
note "import check"
PYTHONPATH=$SRC "$PY" -c "
import torch, transformers, msgpack, websockets
from deployment.model_server.policy_wrapper import PolicyServerWrapper
print('[setup-starvla] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)
"
note "venv size: $(du -sh "$VENV" | cut -f1); models: $(du -sh "$ROOT/models" | cut -f1)"
note "done: source $ROOT/env.sh"
