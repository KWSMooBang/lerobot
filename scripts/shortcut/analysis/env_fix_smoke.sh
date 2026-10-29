#!/usr/bin/env bash
# T009 smoke: pi05lib C0 on the tasks the LIBERO env fix touches, fixed env (+ the unfixed env on LIBERO-10
# for contrast). GPU 0 under flock; writes only under $OUT (not runs/ or results.tsv).
#
#   scripts/shortcut/analysis/env_fix_smoke.sh [out dir] [episodes]
set -euo pipefail
OUT=${1:-/outputs/VLA-shortcut-exps/smoke/T009}
EPS=${2:-5}
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=src:scripts/shortcut
LOCK=""
if [ -d /queue/locks ]; then
  LOCK=/queue/locks/$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0).lock
fi
run() {  # name raw suite tasks
  local cmd=(uv run python scripts/shortcut/run_eval.py --policy pi05 --suite "$3" --condition C0
    --instr-set "scripts/shortcut/instructions/$3.json" --tasks "$4" --episodes "$EPS" --video first:1
    --out "$OUT/$1")
  echo "[smoke] $1 SC_LIBERO_RAW=$2 $(date -Is)"
  if [ -n "$LOCK" ]; then
    SC_LIBERO_RAW=$2 flock "$LOCK" "${cmd[@]}"
  else
    SC_LIBERO_RAW=$2 "${cmd[@]}"
  fi
}
run fix-l10 0 libero_10 0,1,7
run fix-obj 0 libero_object 0,1
run raw-l10 1 libero_10 0,1,7
for d in "$OUT"/*/; do
  uv run python - "$d" <<'EOF'
import json, sys
from collections import defaultdict
eps = [json.loads(line) for line in open(sys.argv[1] + "episodes.jsonl")]
by = defaultdict(list)
for e in eps:
    by[e["task_id"]].append(bool(e.get("success_orig", e.get("success"))))
print(sys.argv[1], {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by.items())},
      f"total {sum(map(sum, by.values()))}/{sum(map(len, by.values()))}")
EOF
done
