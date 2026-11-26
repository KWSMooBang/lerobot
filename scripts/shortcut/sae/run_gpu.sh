#!/usr/bin/env bash
# T034: run one command on GPU 0 under the queue lock and log its wall time (GPU-h ledger).
#   scripts/shortcut/sae/run_gpu.sh <tag> <command ...>
set -uo pipefail
TAG=$1; shift
LOCK=/queue/locks/GPU-b65104ab-c949-afdb-6ebf-6a8af3d8d33b.lock
LEDGER=${T034_LEDGER:-/outputs/VLA-shortcut-exps/sae/crosscoder/gpu_time.tsv}
mkdir -p "$(dirname "$LEDGER")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=SkylakeX
COMMIT=$(git -C "$(dirname "$0")" rev-parse --short HEAD)
exec 9>"$LOCK"
echo "[run_gpu] $TAG waiting for GPU 0 lock" >&2
flock 9
t0=$(date +%s)
echo "[run_gpu] $TAG start $(date '+%F %T') commit $COMMIT" >&2
"$@"
rc=$?
t1=$(date +%s)
printf '%s\t%s\t%s\t%d\t%d\t%s\n' "$TAG" "$(date -d @$t0 '+%F %T')" "$(date -d @$t1 '+%F %T')" $((t1 - t0)) $rc "$COMMIT" >> "$LEDGER"
awk -F'\t' '{s+=$4} END {printf "[run_gpu] GPU-h so far: %.2f\n", s/3600}' "$LEDGER" >&2
exit $rc
