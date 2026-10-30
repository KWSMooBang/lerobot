#!/usr/bin/env bash
# T015 smoke on a server GPU, one command:
#   scripts/shortcut/sae/smoke.sh --out-root /data/sc/smoke/T015 [--gpu 0] [--model pi05lib]
# Every call writes into a new, empty run dir <out-root>/run-<time>-<host>-<rand>/ (printed at the start):
# 1. hook checks on the real π0.5 (smoke.py: (a) action unchanged with hooks, (b) layer L out = L+1 in,
#    (c) padding / empty camera excluded, (d) group-weighted mean = mean over all valid tokens, (e) steering)
# 2. all-layer collection of 3 episodes (goal 379, object 807, spatial 1261 = 365 timesteps) + resume no-op
# 3. SAE on pg05 and ae05: 600 steps, stopped at 200 and resumed, then stats
# 4. SAE step timing at batch 4096
# 5. summary: every step's exit code (steps.tsv) + every sub-check → PASS/FAIL, measured timesteps/s →
#    full-collection hours and storage, SAE hours per layer (summary.json). Exit code 0 only on PASS.
# Not for the shared dev container (no GPU use there).
set -uo pipefail
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
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
S=scripts/shortcut/sae
EPS=379,807,1261
mkdir -p "$OUT" || exit 2
RUN="$OUT/run-$(date +%Y%m%d-%H%M%S)-$(hostname)-$RANDOM"
mkdir "$RUN" || { echo "run dir $RUN exists" >&2; exit 2; }   # never reuse an old run's files
echo "[smoke] run dir $RUN"
: > "$RUN/steps.tsv"

step() {  # step <name> <cmd...>: log to <name>.log, record the command's exit code in steps.tsv
  local name=$1; shift
  echo; echo "=== $name: $*"
  "$@" 2>&1 | tee "$RUN/$name.log"
  local rc=${PIPESTATUS[0]}
  printf '%s\t%s\n' "$name" "$rc" >> "$RUN/steps.tsv"
  [ "$rc" = 0 ] || echo "[smoke] step $name FAILED (exit $rc)"
}

step hook_check uv run python $S/smoke.py --out "$RUN" --model "$MODEL" --episodes $EPS --gpu "$GPU"
step collect uv run python $S/collect.py --model "$MODEL" --out-root "$RUN" --episodes $EPS --parts 3 \
  --gpus "$GPU" --batch 32
step collect_resume uv run python $S/collect.py --model "$MODEL" --out-root "$RUN" --episodes $EPS --parts 3 \
  --gpus "$GPU"
COMMON=(--model "$MODEL" --out-root "$RUN" --gpus "$GPU" --batch 64 --steps 600 --log-every 20 --ckpt-every 100)
step train_part1 uv run python $S/train.py "${COMMON[@]}" --layers pg:5,ae:5 --stop-after 200
step train_resume uv run python $S/train.py "${COMMON[@]}" --layers pg:5,ae:5
step sae_bench uv run python $S/smoke.py --out "$RUN" --bench-sae --gpu "$GPU"
echo; echo "=== summary"
uv run python $S/smoke.py --out "$RUN" --summarize --model "$MODEL"
rc=$?
echo "[smoke] $RUN/summary.json"
exit $rc
