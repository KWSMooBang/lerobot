#!/usr/bin/env bash
# T019: CPU render (mesa EGL, no GPU) of the clean tasks of LIBERO-Plus and LIBERO-PRO in their own venvs, then the
# comparison with the training first frames and with our T009-fixed env (bench_render_compare.py, main venv).
#
#   scripts/shortcut/analysis/bench_render.sh <out dir> [inits, default 0-4]
#
# Venvs: $LIBERO_PLUS_VENV (default <repo>/.venvs/libero_plus), $LIBERO_PRO_PYTHON (<repo>/.venvs/libero_pro/bin/python).
set -euo pipefail
OUT=$1
INITS=${2:-0-4}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PLUS_PY=${LIBERO_PLUS_VENV:-$REPO/.venvs/libero_plus}/bin/python
PRO_PY=${LIBERO_PRO_PYTHON:-$REPO/.venvs/libero_pro/bin/python}
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2
export __EGL_VENDOR_LIBRARY_FILENAMES=${MESA_EGL_JSON:-/usr/share/glvnd/egl_vendor.d/50_mesa.json}
mkdir -p "$OUT/logs"
pids=()
for s in libero_spatial libero_object libero_goal libero_10; do
  "$PLUS_PY" "$REPO/scripts/shortcut/analysis/bench_render.py" --bench plus --suite "$s" --inits "$INITS" \
    --cache "$OUT/plus" > "$OUT/logs/plus_$s.log" 2>&1 & pids+=($!)
  "$PRO_PY" "$REPO/scripts/shortcut/analysis/bench_render.py" --bench pro --suite "$s" --inits "$INITS" \
    --cache "$OUT/pro" > "$OUT/logs/pro_$s.log" 2>&1 & pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "a render failed, see $OUT/logs" >&2; exit 1; }
uv run --project "$REPO" python "$REPO/scripts/shortcut/analysis/bench_render_compare.py" --renders "$OUT" --out "$OUT"
