#!/usr/bin/env bash
# T015 smoke on a server GPU, one command:
#   scripts/shortcut/sae/smoke.sh --out-root /data/sc/smoke/T015 [--gpu 0] [--model pi05lib]
# 1. hook checks on the real π0.5 (smoke.py: (a) action unchanged with hooks, (b) layer L out = L+1 in,
#    (c) padding / empty camera excluded, (d) group-weighted mean = mean over all valid tokens, (e) steering)
# 2. all-layer collection of 3 episodes (goal 379, object 807, spatial 1261 = 365 timesteps) + resume no-op
# 3. SAE on pg05 and ae05: 600 steps, stopped at 200 and resumed, then stats
# 4. SAE step timing at batch 4096
# 5. summary: PASS/FAIL per check, measured timesteps/s → full-collection hours and storage, SAE hours per layer
#    (<out-root>/summary.json). Not for the shared dev container (no GPU use there).
set -euo pipefail
OUT=""; GPU=0; MODEL=pi05lib
while [ $# -gt 0 ]; do
  case "$1" in
    --out-root) OUT=$2; shift ;;
    --gpu) GPU=$2; shift ;;
    --model) MODEL=$2; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$OUT" ] || { echo "usage: $0 --out-root DIR [--gpu N] [--model pi05lib]" >&2; exit 2; }
if [ -d /queue/locks ] && [ "${SC_SMOKE_ALLOW_HERE:-0}" != 1 ]; then
  echo "this looks like the shared dev container (/queue/locks): GPU smoke runs on the server only" >&2
  exit 2
fi
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
S=scripts/shortcut/sae
EPS=379,807,1261
mkdir -p "$OUT"
step() { echo; echo "=== $*"; }
set +e
step "1/4 hook checks"
uv run python $S/smoke.py --out "$OUT" --model "$MODEL" --episodes $EPS --gpu "$GPU" 2>&1 | tee "$OUT/hook_check.log"
step "2/4 collection (3 episodes, all layers) + resume"
uv run python $S/collect.py --model "$MODEL" --out-root "$OUT" --episodes $EPS --parts 3 --gpus "$GPU" \
  --batch 32 2>&1 | tee "$OUT/collect.log"
uv run python $S/collect.py --model "$MODEL" --out-root "$OUT" --episodes $EPS --parts 3 --gpus "$GPU" \
  2>&1 | tee "$OUT/collect_resume.log"
step "3/4 SAE pg05 + ae05 (stop at 200, resume to 600) + stats"
COMMON=(--model "$MODEL" --out-root "$OUT" --gpus "$GPU" --batch 64 --steps 600 --log-every 20 --ckpt-every 100)
uv run python $S/train.py "${COMMON[@]}" --layers pg:5,ae:5 --stop-after 200 2>&1 | tee "$OUT/train_part1.log"
uv run python $S/train.py "${COMMON[@]}" --layers pg:5,ae:5 2>&1 | tee "$OUT/train_resume.log"
step "4/4 SAE step timing (batch 4096)"
uv run python $S/smoke.py --out "$OUT" --bench-sae --gpu "$GPU" 2>&1 | tee "$OUT/sae_bench.log"
set -e
step "summary"
uv run python $S/smoke.py --out "$OUT" --summarize --model "$MODEL"
