#!/usr/bin/env bash
# Multi-GPU shortcut evals (parallel/run_multi.py): all runs of one call share a work queue over
# G GPUs x P workers x N envs. Same run_ids and run dir layout as submit.sh.
#
#   scripts/shortcut/submit_multi.sh [--dry-run] --ckpt pi05lib,pi05base --suites goal,spat \
#       --conds C0,C1,C4 [--repeats 0,1,2] [--episodes 50] [--tasks 0-9] [--video first:5] \
#       [--id-prefix sc] [--gpus 0,1,2,3] [--procs-per-gpu auto] [--num-envs auto] \
#       [--out-root ./outputs/VLA-shortcut-exps] [-- <extra run_multi.py args, e.g. --compile>]
#
# run_id: <prefix>-<ckpt tag>-<suite short>-<cond>[-r<k>] under <out-root>/runs/ (as submit.sh).
# Runs in the foreground (use tmux/nohup); stdout is also written to <out-root>/multi/<session>/.
# Each finished run gets STATUS, ended_at and one line in <out-root>/results.tsv (submit.sh columns).
# No /harness or /queue needed; on servers that have /queue/locks the GPUs are flock'ed for the session.
# Rerunning the same command resumes: finished episodes are skipped.
# --dry-run: units per run, expected env steps and estimated wall time (parallel/README.md).
set -euo pipefail

CKPTS=pi05lib; SUITES=goal; CONDS=C0; REPEATS=0; EPISODES=50; TASKS=0-9; VIDEO=; PREFIX=sc
GPUS=0; PROCS=auto; NENVS=auto; OUT_ROOT=./outputs/VLA-shortcut-exps; DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --ckpt) CKPTS=$2; shift ;;
    --suites) SUITES=$2; shift ;;
    --conds) CONDS=$2; shift ;;
    --repeats) REPEATS=$2; shift ;;
    --episodes) EPISODES=$2; shift ;;
    --tasks) TASKS=$2; shift ;;
    --video) VIDEO=$2; shift ;;
    --id-prefix) PREFIX=$2; shift ;;
    --gpus) GPUS=$2; shift ;;
    --procs-per-gpu) PROCS=$2; shift ;;
    --num-envs) NENVS=$2; shift ;;
    --out-root) OUT_ROOT=$2; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

REPO=$(git rev-parse --show-toplevel)
mkdir -p "$OUT_ROOT"
OUT_ROOT=$(cd "$OUT_ROOT" && pwd)
cmd=(uv run --project "$REPO" python "$REPO/scripts/shortcut/parallel/run_multi.py"
     --ckpt "$CKPTS" --suites "$SUITES" --conds "$CONDS" --repeats "$REPEATS" --episodes "$EPISODES"
     --tasks "$TASKS" --id-prefix "$PREFIX" --gpus "$GPUS" --procs-per-gpu "$PROCS" --num-envs "$NENVS"
     --out-root "$OUT_ROOT")
[ -n "$VIDEO" ] && cmd+=(--video "$VIDEO")
[ "$DRY" = 1 ] && cmd+=(--dry-run)
cmd+=(${EXTRA[@]+"${EXTRA[@]}"})
echo "[multi] ${cmd[*]}"
exec "${cmd[@]}"
