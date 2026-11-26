#!/usr/bin/env bash
# T034 runs 3–5 on GPU 1 (human approval 2026-11-26 via lead; GPU 0 not used again):
# resume the VLM collection → VLM cache + checks (CPU) → VLM k 64 / 128 at n 8 192 → joint (PREREG Addendum D order).
set -uo pipefail
cd "$(dirname "$0")/../../.."
SAE=scripts/shortcut/sae
LOG=/outputs/VLA-shortcut-exps/sae/crosscoder/logs
PY=$PWD/.venv312/bin/python
export OPENBLAS_CORETYPE=SkylakeX T034_GPU=1
# ---- run 3: collection (resumes; finished parts kept). collect.py flocks GPU 1 itself.
(
  cd $SAE
  export SC_LIBERO_DATA_ROOT=/data0/woosung/datasets/VLA-shortcut-exps/libero/hf_libero
  export HF_DATASETS_CACHE=/data0/woosung/datasets/VLA-shortcut-exps/libero/hf_cache HF_HUB_OFFLINE=1
  T034_NOLOCK=1 ./run_gpu.sh collect_vlm $PY collect.py --model pi05lib --layers emb,pg:0-17 --frame-stride 2 \
    --acts-dir /data0/woosung/datasets/VLA-shortcut-exps/sae/vlm_acts/pi05lib_s2 --gpus 1 --allow-any-gpu \
    --batch 32 --workers 6
) > $LOG/collect_vlm_gpu1.log 2>&1 || { echo "collection failed"; exit 1; }
n=$(ls /data0/woosung/datasets/VLA-shortcut-exps/sae/vlm_acts/pi05lib_s2/parts | grep -c "^ep[0-9-]*$")
[ "$n" -eq 64 ] || { echo "collection incomplete: $n/64 parts"; exit 1; }
du -sh /data0/woosung/datasets/VLA-shortcut-exps/sae/vlm_acts/pi05lib_s2
$PY $SAE/cc_data.py vlm > $LOG/cache_vlm.log 2>&1 || { echo "vlm cache failed"; exit 1; }
$PY $SAE/cc_check_vlm.py > $LOG/check_vlm.log 2>&1 || { echo "vlm check FAILED (see check_vlm.log); stop"; exit 1; }
echo "check passed $(date '+%F %T')"
# ---- run 4: VLM, k ∈ {64, 128} (PREREG §3 / Addendum B; Addendum C's one-run cut is lifted with GPU 1)
for k in 64 128; do
  $SAE/run_gpu.sh vlm_n8192_k$k $PY $SAE/cc_train.py --data vlm --name vlm_n8192_k$k --n-latents 8192 --k $k \
    > $LOG/vlm_n8192_k$k.log 2>&1 || echo "vlm_n8192_k$k failed"
done
K=$($PY - <<'PYEOF'
import json
r = {k: json.load(open(f"/outputs/VLA-shortcut-exps/sae/crosscoder/runs/vlm_n8192_k{k}.json"))["heldout"]["ve_mean"] for k in (64, 128)}
print(128 if r[128] - r[64] >= 0.01 else 64)
PYEOF
)
echo "vlm k = $K $(date '+%F %T')"
# ---- run 5: joint (Addendum D: fp32 Adam n 8 192 first; then Addendum B fallbacks on OOM)
try() {  # name, extra args
  local name=$1; shift
  $SAE/run_gpu.sh $name $PY $SAE/cc_train.py --data joint --name $name --k $K "$@" > $LOG/$name.log 2>&1 && {
    echo "joint done $name"; exit 0; }
  grep -q "OutOfMemoryError" $LOG/$name.log || { echo "joint $name failed (not OOM)"; exit 1; }
  echo "joint $name OOM"
}
try joint_n8192_k$K --n-latents 8192
try joint_n8192_k${K}_lowmem --n-latents 8192 --low-mem-adam
try joint_n8192_k${K}_lowmem_b2048 --n-latents 8192 --low-mem-adam --batch 2048
try joint_n4096_k$K --n-latents 4096
echo "joint failed"
