#!/usr/bin/env bash
# T018 language probe on a GPU server: probe.py (one policy server at a time, sc-bench's launcher) → measure.py (CPU).
#
#   scripts/shortcut/analysis/langprobe/run_probe.sh \
#       --models pi05droid,pi05lib,pi05base,starvla-qwen3pi-libero --out-root /outputs/VLA-shortcut-exps --gpu 0
#   ... --dry-run [--est-ms pi05lib=180,pi05droid=200]    forwards × ms/forward → GPU-h (no server, no GPU)
#
# Options: --models (required), --out-root (default $SC_OUT_ROOT or /outputs/VLA-shortcut-exps), --gpu (default 0),
#   --run-id (default probe → <out-root>/analysis/langprobe/<run-id>/), --obs-dir (default
#   <out-root>/analysis/langprobe/obs: the files made by make_obs.py in the dev container; sha256 checked against
#   manifest.json), --sets (default A,B), --skip-measure, --measure-only, --dry-run, --est-ms; anything after `--`
#   goes to probe.py (e.g. -- --tasks 1,8 --max-obs-per-task 1 for a smoke run under a new --run-id).
# Same command again = resume (done queries are skipped; measure re-executes only models whose probe changed).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT_ROOT=${SC_OUT_ROOT:-/outputs/VLA-shortcut-exps}
MODELS="" GPU=0 RUN_ID=probe OBS_DIR="" SETS=A,B EST_MS="" DRY=0 SKIP_MEASURE=0 MEASURE_ONLY=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS=$2; shift 2 ;;
    --out-root) OUT_ROOT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    --run-id) RUN_ID=$2; shift 2 ;;
    --obs-dir) OBS_DIR=$2; shift 2 ;;
    --sets) SETS=$2; shift 2 ;;
    --est-ms) EST_MS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --skip-measure) SKIP_MEASURE=1; shift ;;
    --measure-only) MEASURE_ONLY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "run_probe.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODELS" ] || { echo "run_probe.sh: --models is required" >&2; exit 2; }
OBS_DIR=${OBS_DIR:-$OUT_ROOT/analysis/langprobe/obs}
RUN_DIR=$OUT_ROOT/analysis/langprobe/$RUN_ID
if [ ! -f "$OBS_DIR/manifest.json" ]; then
  echo "run_probe.sh: $OBS_DIR/manifest.json missing: copy the obs dir made by make_obs.py (see README.md)" >&2
  exit 2
fi
PY=(uv run --project "$REPO" --extra shortcut-bench python)
PROBE=("${PY[@]}" "$HERE/probe.py" --models "$MODELS" --obs-dir "$OBS_DIR" --sets "$SETS" --out "$RUN_DIR"
       --out-root "$OUT_ROOT" --gpu "$GPU")
if [ "$DRY" = 1 ]; then
  exec "${PROBE[@]}" --dry-run ${EST_MS:+--est-ms "$EST_MS"} "${EXTRA[@]}"
fi
mkdir -p "$RUN_DIR"
{
  echo "$(date -Iseconds) $0 --models $MODELS --out-root $OUT_ROOT --gpu $GPU --run-id $RUN_ID ${EXTRA[*]}"
  echo "git $(git -C "$REPO" rev-parse HEAD) dirty=$(git -C "$REPO" status --porcelain -- scripts src | wc -l)"
} >> "$RUN_DIR/run_probe.log"
if [ "$MEASURE_ONLY" = 0 ]; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID "${PROBE[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$RUN_DIR/run_probe.log"
fi
if [ "$SKIP_MEASURE" = 0 ]; then
  # physics only (cameras off); EGL on the server's GPU just to create the render context LIBERO wants
  MUJOCO_GL=${MUJOCO_GL:-egl} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$GPU} CUDA_VISIBLE_DEVICES=$GPU \
    "${PY[@]}" "$HERE/measure.py" --run "$RUN_DIR" --obs-dir "$OBS_DIR" \
    2>&1 | tee -a "$RUN_DIR/run_probe.log"
  echo "summary: $RUN_DIR/measure/summary.md"
fi
