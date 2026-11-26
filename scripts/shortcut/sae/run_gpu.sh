#!/usr/bin/env bash
# T034: run one command on GPU 0 under the queue lock and log its wall time (GPU-h ledger).
#   scripts/shortcut/sae/run_gpu.sh <tag> <command ...>
set -uo pipefail
TAG=$1; shift
GPU=${T034_GPU:-0}  # 2026-11-26 17:55: human approved GPU 1 (via lead); runs after the expert sweep use T034_GPU=1
case $GPU in
  0) LOCK=/queue/locks/GPU-b65104ab-c949-afdb-6ebf-6a8af3d8d33b.lock ;;
  1) LOCK=/queue/locks/GPU-464336e4-7e13-f6c4-5951-2221e727ef58.lock ;;
  *) echo "T034 uses GPU 0 or 1 only" >&2; exit 2 ;;
esac
LEDGER=${T034_LEDGER:-/outputs/VLA-shortcut-exps/sae/crosscoder/gpu_time.tsv}
mkdir -p "$(dirname "$LEDGER")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OPENBLAS_CORETYPE=SkylakeX
TAG="${TAG}_gpu$GPU"
COMMIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)
if [ -z "${T034_NOLOCK:-}" ]; then  # collect.py flocks the GPU itself (sc_train.acquire_gpu_locks)
  exec 9>"$LOCK"
  echo "[run_gpu] $TAG waiting for GPU 0 lock" >&2
  flock 9
fi
t0=$(date +%s)
echo "[run_gpu] $TAG start $(date '+%F %T') commit $COMMIT" >&2
"$@"
rc=$?
t1=$(date +%s)
printf '%s\t%s\t%s\t%d\t%d\t%s\n' "$TAG" "$(date -d @$t0 '+%F %T')" "$(date -d @$t1 '+%F %T')" $((t1 - t0)) $rc "$COMMIT" >> "$LEDGER"
awk -F'\t' '{s+=$4} END {printf "[run_gpu] GPU-h so far: %.2f\n", s/3600}' "$LEDGER" >&2
exit $rc
