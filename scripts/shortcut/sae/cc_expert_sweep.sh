#!/usr/bin/env bash
# T034 runs 1–2: real-data smoke (2 % of episodes, tiny crosscoder) then the expert size sweep (PREREG §3).
set -uo pipefail
cd "$(dirname "$0")/../../.."
SAE=scripts/shortcut/sae
LOG=/outputs/VLA-shortcut-exps/sae/crosscoder/logs
mkdir -p $LOG
PY="uv run --no-sync python"
$SAE/run_gpu.sh smoke_data $PY $SAE/cc_train.py --data expert --name smoke_expert_frac02 --n-latents 2048 --k 32 \
  --steps 300 --frac 0.02 --weights /data0/woosung/datasets/VLA-shortcut-exps/cache/t034_smoke \
  --out /outputs/VLA-shortcut-exps/sae/crosscoder/smoke > $LOG/smoke_data.log 2>&1 || { echo "smoke failed"; exit 1; }
for nk in "8192 64" "8192 128" "16384 64" "16384 128"; do
  set -- $nk
  name=expert_n$1_k$2
  $SAE/run_gpu.sh $name $PY $SAE/cc_train.py --data expert --name $name --n-latents $1 --k $2 > $LOG/$name.log 2>&1 \
    || echo "$name failed"
done
echo "sweep done"
