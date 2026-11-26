#!/usr/bin/env bash
# T034 runs 4–5 after the collection: VLM cache + checks (CPU) → VLM sweep k ∈ {64, 128} at n 8 192 → joint
# (PREREG §3 + Addendum B). Stops on a failed check.
set -uo pipefail
cd "$(dirname "$0")/../../.."
SAE=scripts/shortcut/sae
LOG=/outputs/VLA-shortcut-exps/sae/crosscoder/logs
PY=.venv312/bin/python
export OPENBLAS_CORETYPE=SkylakeX
until grep -q "collect exit" $LOG/collect_vlm.log 2>/dev/null; do sleep 60; done
grep -q "collect exit 0" $LOG/collect_vlm.log || { echo "collection failed"; exit 1; }
$PY $SAE/cc_data.py vlm > $LOG/cache_vlm.log 2>&1 || { echo "vlm cache failed"; exit 1; }
$PY $SAE/cc_check_vlm.py > $LOG/check_vlm.log 2>&1 || { echo "vlm check FAILED (see check_vlm.log); stop"; exit 1; }
echo "check passed"
for k in 64 128; do
  $SAE/run_gpu.sh vlm_n8192_k$k $PY $SAE/cc_train.py --data vlm --name vlm_n8192_k$k --n-latents 8192 --k $k \
    > $LOG/vlm_n8192_k$k.log 2>&1 || echo "vlm_n8192_k$k failed"
done
K=$($PY - <<'EOF'
import json
r = {k: json.load(open(f"/outputs/VLA-shortcut-exps/sae/crosscoder/runs/vlm_n8192_k{k}.json"))["heldout"]["ve_mean"] for k in (64, 128)}
print(128 if r[128] - r[64] >= 0.01 else 64)
EOF
)
echo "vlm k = $K"
name=joint_n8192_k$K
$SAE/run_gpu.sh $name $PY $SAE/cc_train.py --data joint --name $name --n-latents 8192 --k $K --low-mem-adam \
  > $LOG/$name.log 2>&1 && { echo "joint done $name"; exit 0; }
grep -q "OutOfMemoryError" $LOG/$name.log || { echo "joint failed (not OOM)"; exit 1; }
name=joint_n8192_k${K}_b2048
$SAE/run_gpu.sh $name $PY $SAE/cc_train.py --data joint --name $name --n-latents 8192 --k $K --low-mem-adam --batch 2048 \
  > $LOG/$name.log 2>&1 && { echo "joint done $name"; exit 0; }
grep -q "OutOfMemoryError" $LOG/$name.log || { echo "joint failed (not OOM)"; exit 1; }
name=joint_n4096_k$K
$SAE/run_gpu.sh $name $PY $SAE/cc_train.py --data joint --name $name --n-latents 4096 --k $K \
  > $LOG/$name.log 2>&1 && echo "joint done $name" || echo "joint failed"
