#!/usr/bin/env bash
# Launcher sc-bench uses for registry rows with `server=droid` (bench/core/runner.py): runs server.py in the lerobot
# venv (the one sc-bench runs in: $SC_PYTHON, set by the runner; else `uv run` in this repo). Arguments are
# server.py's (--ckpt, --port, --ready-file, ...).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../../../.." && pwd)
if [ -n "${SC_PYTHON:-}" ]; then
  exec "$SC_PYTHON" "$HERE/server.py" "$@"
fi
exec uv run --project "$REPO" --extra shortcut-bench python "$HERE/server.py" "$@"
