#!/usr/bin/env bash
# Render-fixed view of an existing LIBERO-PRO install without touching it (T023).
#
# An install made before T019 (e.g. the T013 one) has the shipped sRGB textures and no fix marker. Both forks
# load assets from their package dir, so the fix needs a package copy: `render_fix.py mirror` (symlink farm of
# libero/libero, as T019's verification renders) + `render_fix.py apply` on the copy, and a python wrapper that
# puts the copy first on PYTHONPATH. The original install is only read.
#
#   scripts/shortcut/bench/libero_pro/fixed_mirror.sh <existing .venvs dir> <new dir>
#   export SC_BENCH_VENVS=<existing .venvs dir> LIBERO_PRO_ROOT=<new dir>/src/libero/libero \
#          LIBERO_PRO_PYTHON=<new dir>/python
# bench_version then ends in /fix-fix1-<hash> (catalog.render_fix_tag reads the copy's marker).
set -euo pipefail
SRC_VENVS=$(cd "$1" && pwd)
DST=$2
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=$SRC_VENVS/libero_pro/bin/python
SRC=$SRC_VENVS/libero_pro_src/libero
mkdir -p "$DST/src/libero"
DST=$(cd "$DST" && pwd)
for d in "$SRC"/*; do
  n=$(basename "$d")
  [ "$n" = libero ] && continue
  [ -e "$DST/src/libero/$n" ] || ln -s "$d" "$DST/src/libero/$n"
done
# bench_version stamp (catalog.bench_version reads <root>/../../SC_VERSION)
[ -e "$DST/src/SC_VERSION" ] || ln -s "$SRC_VENVS/libero_pro_src/SC_VERSION" "$DST/src/SC_VERSION"
"$PY" "$HERE/../render_fix.py" mirror "$SRC/libero" "$DST/src/libero/libero"
"$PY" "$HERE/../render_fix.py" apply --assets "$DST/src/libero/libero/assets"
cat > "$DST/python" <<EOF
#!/usr/bin/env bash
# LIBERO-PRO venv python with the render-fixed package copy first on the path (fixed_mirror.sh)
export PYTHONPATH="$DST/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$PY" "\$@"
EOF
chmod +x "$DST/python"
"$DST/python" -c "import libero.libero as m, sys; print('libero.libero from', m.__file__)"
"$PY" "$HERE/../render_fix.py" status --assets "$DST/src/libero/libero/assets"
