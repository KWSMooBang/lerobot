#!/usr/bin/env bash
# GPU smoke for the StarVLA server (T016). Run on the GPU server after setup_server.sh (lerobot venv, LIBERO) and
# setup_starvla.sh (StarVLA venv + checkpoint):
#
#   scripts/shortcut/serve/external/starvla/smoke.sh --out-root /outputs/VLA-shortcut-exps [--gpu 0] [--starvla-root DIR]
#
# (a) action equality on real LIBERO renders: dump_frames.py (lerobot venv, the `ours` env) for libero_goal task 1
#     and libero_spatial task 0 → compare.py (StarVLA venv, GPU, bf16): our server's actions == StarVLA's own
#     WebsocketPolicyServer + LIBERO ModelClient actions with the same noise (exact, 2 chunks × 8 actions each).
# (b) sc-bench --ckpt starvla-qwen3pi-libero, libero_goal C0 × tasks 1,8 × 2 inits, to the end (run id
#     smoke-T016-starvla under <out-root>/smoke/T016, not runs/); s/episode measured from the episode records.
# Prints PASS/FAIL per part; exit 0 iff both pass. Outputs: <out-root>/smoke/T016/.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
OUT_ROOT=${SC_OUT_ROOT:-/outputs/VLA-shortcut-exps}
GPU=${CUDA_VISIBLE_DEVICES:-0}
STARVLA_ROOT_ARG=
while [ $# -gt 0 ]; do
  case "$1" in
    --out-root) OUT_ROOT=$2; shift ;;
    --gpu) GPU=$2; shift ;;
    --starvla-root) STARVLA_ROOT_ARG=$2; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
export STARVLA_ROOT=${STARVLA_ROOT_ARG:-${STARVLA_ROOT:-$OUT_ROOT/starvla}}
[ -f "$STARVLA_ROOT/env.sh" ] || { echo "missing $STARVLA_ROOT/env.sh: run setup_starvla.sh first" >&2; exit 2; }
# shellcheck disable=SC1091
source "$STARVLA_ROOT/env.sh"
SMOKE=$OUT_ROOT/smoke/T016
mkdir -p "$SMOKE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU
LEROBOT=(uv run --project "$REPO" --extra shortcut-bench python)
ok_a=FAIL; ok_b=FAIL

echo "[smoke] (a) frames from the ours LIBERO env"
"${LEROBOT[@]}" "$HERE/dump_frames.py" --source libero-env --suite libero_goal --task 1 --out "$SMOKE/frames_goal_t1.npz"
"${LEROBOT[@]}" "$HERE/dump_frames.py" --source libero-env --suite libero_spatial --task 0 --out "$SMOKE/frames_spatial_t0.npz"
echo "[smoke] (a) compare (StarVLA venv, cuda, bf16)"
if PYTHONPATH="$STARVLA_SRC" "$STARVLA_PYTHON" "$HERE/compare.py" --device cuda \
  --frames "$SMOKE/frames_goal_t1.npz,$SMOKE/frames_spatial_t0.npz" --out "$SMOKE/compare.json" \
  2>&1 | tee "$SMOKE/compare.log"; then
  ok_a=PASS
fi

echo "[smoke] (b) sc-bench libero_goal C0 × tasks 1,8 × 2 inits"
RUN=smoke-T016-starvla
tic=$(date +%s)
if "$REPO/scripts/shortcut/bench/sc-bench" --ckpt starvla-qwen3pi-libero --battery smoke --suites libero_goal \
  --only C0 --tasks 1,8 --episodes 2 --gpus "$GPU" --out-root "$SMOKE" --run-id "$RUN" 2>&1 | tee "$SMOKE/bench.log"; then
  ok_b=PASS
fi
wall=$(($(date +%s) - tic))
python3 - "$SMOKE/runs/$RUN" "$wall" "$ok_b" <<'EOF' | tee "$SMOKE/bench_summary.txt"
import json, sys
from pathlib import Path

run, wall, ok = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
eps = [json.loads(x) for f in run.glob("jobs/*/episodes.jsonl") for x in f.read_text().splitlines() if x.strip()]
n = len(eps)
ident = {}
for f in run.glob("jobs/*/run_meta.json"):
    ident = json.loads(f.read_text())["identity"]
s_ep = sum(e.get("wall_time_s", 0) for e in eps) / max(n, 1)
steps = sum(e.get("n_steps", 0) for e in eps) / max(n, 1)
sr = sum(bool(e["success_orig"]) for e in eps) / max(n, 1)
if n != 4:
    ok = "FAIL"
print(f"[smoke] (b) {ok}: {n}/4 episodes, C0 SR {sr:.2f}, {s_ep:.1f} s/episode ({steps:.0f} env steps/episode), "
      f"wall {wall} s incl. model load")
print("[smoke] (b) identity: " + json.dumps({k: ident.get(k) for k in (
    "policy", "camera", "starvla_sha", "ckpt_sha", "base_vlm_sha", "n_action_steps", "max_steps", "exec_mode", "gpu")}))
EOF
grep -q "(b) PASS" "$SMOKE/bench_summary.txt" || ok_b=FAIL
echo "[smoke] (a) action equality: $ok_a   (b) sc-bench C0 run: $ok_b   -> $SMOKE"
[ "$ok_a" = PASS ] && [ "$ok_b" = PASS ]
