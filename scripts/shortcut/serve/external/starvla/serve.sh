#!/usr/bin/env bash
# Launcher sc-bench uses for registry rows with `server=starvla` (bench/core/runner.py): runs server.py with the
# StarVLA venv. Python: $STARVLA_PYTHON, else from $STARVLA_ROOT/env.sh (default $SC_OUT_ROOT/starvla/env.sh,
# written by setup_starvla.sh). Arguments are server.py's (--ckpt, --port, --ready-file, ...).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${STARVLA_PYTHON:-}" ]; then
  ENV_SH=${STARVLA_ROOT:-${SC_OUT_ROOT:-/outputs/VLA-shortcut-exps}/starvla}/env.sh
  if [ ! -f "$ENV_SH" ]; then
    echo "[starvla] $ENV_SH missing: run $HERE/setup_starvla.sh (or set STARVLA_ROOT / STARVLA_PYTHON)" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$ENV_SH"
fi
exec "$STARVLA_PYTHON" "$HERE/server.py" "$@"
