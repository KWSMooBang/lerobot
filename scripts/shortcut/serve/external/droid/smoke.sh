#!/usr/bin/env bash
# GPU smoke for π0.5-DROID zero-shot on LIBERO (T017). Run on the GPU server after setup_server.sh (lerobot venv,
# LIBERO, PaliGemma tokenizer):
#
#   scripts/shortcut/serve/external/droid/smoke.sh --out-root /outputs/VLA-shortcut-exps [--gpu 0]
#
# sc-bench --ckpt pi05droid --battery droid_smoke: 4 suites × C0 × tasks 0-2 × 3 inits = 36 episodes, DROID
# joint-velocity env (15 Hz), first init of each task on video. Run id smoke-T017-droid under
# <out-root>/smoke/T017 (not runs/). Budget: ≈ 15 min on one RTX 4090 incl. model load (planner: 12,780 env steps).
# PASS = 36 episodes written and every executed action finite and in [-1, 1] (gripper ±1). The C0 success rate is a
# measurement, not a criterion. Prints C0 SR per suite × task, s/episode, video paths, and the GPU-h estimate of the
# main battery (droid_main, P1: Spatial·Goal × C0/C1/C4/C5) from the measured time per env step.
# --mock: the same script on CPU with the mock model + mock env (plumbing check, no GPU, no checkpoint).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
OUT_ROOT=${SC_OUT_ROOT:-/outputs/VLA-shortcut-exps}
GPU=${CUDA_VISIBLE_DEVICES:-0}
MOCK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out-root) OUT_ROOT=$2; shift ;;
    --gpu) GPU=$2; shift ;;
    --mock) MOCK=1 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
SMOKE=$OUT_ROOT/smoke/T017
RUN=smoke-T017-droid
mkdir -p "$SMOKE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID MUJOCO_GL=egl
SCB=("$REPO/scripts/shortcut/bench/sc-bench" --ckpt pi05droid --out-root "$SMOKE" --clients-per-gpu 1)
if [ "$MOCK" = 1 ]; then
  SMOKE=$OUT_ROOT/smoke/T017-mock
  mkdir -p "$SMOKE"
  SCB=("$REPO/scripts/shortcut/bench/sc-bench" --ckpt mock/pi05droid --backend mock --server droid --out-root "$SMOKE")
fi

echo "[smoke] sc-bench --ckpt pi05droid --battery droid_smoke on GPU $GPU"
tic=$(date +%s)
ok=PASS
"${SCB[@]}" --battery droid_smoke --gpus "$GPU" --run-id "$RUN" 2>&1 | tee "$SMOKE/bench.log" || ok=FAIL
wall=$(($(date +%s) - tic))
"${SCB[@]}" --battery droid_main --dry-run --run-id main-plan 2>/dev/null | grep -E "^(ours-|total)" > "$SMOKE/main_plan.txt" || true

uv run --project "$REPO" --extra shortcut-bench python - "$SMOKE/runs/$RUN" "$wall" "$ok" "$SMOKE/main_plan.txt" <<'EOF' | tee "$SMOKE/summary.txt"
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

run, wall, ok, plan = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
eps, bad = [], []
for f in sorted(run.glob("jobs/*/episodes.jsonl")):
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        eps.append(e)
        a = np.load(f.parent / e["steps_file"])["action"]
        if a.ndim != 2 or a.shape[1] != 8 or not np.all(np.isfinite(a)) or np.abs(a).max() > 1 or not np.all(np.isin(a[:, 7], (-1, 1))):
            bad.append(e["steps_file"])
n = len(eps)
if n != 36 or bad:
    ok = "FAIL"
ident = {}
for f in run.glob("jobs/*/run_meta.json"):
    ident = json.loads(f.read_text())["identity"]
by = defaultdict(list)
for e in eps:
    by[(e["suite"], e["task_id"])].append(e)
print(f"[smoke] {ok}: {n}/36 episodes, {len(bad)} with non-finite / out-of-range actions; wall {wall} s incl. model load")
print("[smoke] C0 success per suite × task (3 inits each):")
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    cells = [(t, by[(suite, t)]) for t in range(3)]
    sr = [sum(bool(e["success_orig"]) for e in es) for _, es in cells]
    tot = sum(len(es) for _, es in cells)
    print(f"  {suite:15s} " + "  ".join(f"t{t}: {s}/{len(es)}" for (t, es), s in zip(cells, sr)) + f"   suite {sum(sr)}/{tot}")
sr_all = sum(bool(e["success_orig"]) for e in eps) / max(n, 1)
s_ep = sum(e.get("wall_time_s", 0) for e in eps) / max(n, 1)
steps = sum(e.get("n_steps", 0) for e in eps)
s_step = sum(e.get("wall_time_s", 0) for e in eps) / max(steps, 1)
print(f"[smoke] C0 SR all {sr_all:.3f}; {s_ep:.1f} s/episode, {steps / max(n, 1):.0f} env steps/episode, {1000 * s_step:.1f} ms per env step (env + policy)")
print("[smoke] videos (first init of each task):")
for e in eps:
    if e.get("video"):
        print(f"  {e['suite']} t{e['task_id']} i{e['init_idx']}: {run / 'jobs' / ('ours-' + {'libero_spatial': 'spat', 'libero_object': 'obj', 'libero_goal': 'goal', 'libero_10': 'l10'}[e['suite']] + '-C0') / e['video']}")
print("[smoke] identity: " + json.dumps({k: ident.get(k) for k in (
    "policy", "checkpoint", "ckpt_sha", "stats_sha256", "action_space", "control_freq", "max_joint_delta",
    "joint_kp", "camera", "dtype", "n_action_steps", "max_steps", "exec_mode", "gpu")}))
tot_line = [x.split() for x in plan.read_text().splitlines() if x.startswith("total")] if plan.exists() else []
if tot_line:
    n_main, est_steps = int(tot_line[0][2]), float(tot_line[0][4].replace(",", ""))
    print(f"[smoke] main battery droid_main: {n_main} episodes, planner {est_steps:,.0f} env steps (full-length episodes)")
    print(f"[smoke] main GPU-h ≈ env steps × s/step / 3600 = {est_steps:,.0f} × {s_step:.4f} / 3600 = {est_steps * s_step / 3600:.2f} GPU-h "
          "(1 client per GPU, exec_mode single; wall = GPU-h / #GPUs)")
EOF
grep -q "\] PASS:" "$SMOKE/summary.txt" || ok=FAIL
echo "[smoke] T017 droid smoke: $ok -> $SMOKE"
[ "$ok" = PASS ]
