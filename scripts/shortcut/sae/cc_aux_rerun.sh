#!/usr/bin/env bash
# T034 Addendum E: AuxK 1/8 sensitivity run of the selected expert size, after the joint run, if ≥ 0.5 GPU-h remain.
set -uo pipefail
cd "$(dirname "$0")/../../.."
SAE=scripts/shortcut/sae
LOG=/outputs/VLA-shortcut-exps/sae/crosscoder/logs
until grep -q "joint done\|joint failed\|joint .* failed\|stop" $LOG/gpu1_chain.log 2>/dev/null; do sleep 60; done
used=$(awk -F'\t' '{s+=$4} END {print s+0}' /outputs/VLA-shortcut-exps/sae/crosscoder/gpu_time.tsv)
if [ "$used" -gt $((28800 - 1800)) ]; then echo "skip: ${used}s used"; exit 0; fi
T034_GPU=1 $SAE/run_gpu.sh expert_n8192_k128_aux8 .venv312/bin/python $SAE/cc_train.py --data expert \
  --name expert_n8192_k128_aux8 --n-latents 8192 --k 128 --aux-coef 0.125 > $LOG/expert_n8192_k128_aux8.log 2>&1
echo "aux rerun exit $?"
