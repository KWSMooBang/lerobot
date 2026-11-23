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
SRC_REAL=$(realpath "$SRC")
die() { echo "fixed_mirror.sh: refusing: $*" >&2; exit 1; }
# nothing is created before DST is known to resolve outside the original package
case "$(realpath -m "$DST")/" in "$SRC_REAL"/* | "$(realpath "$SRC_VENVS/libero_pro_src")/") die "$DST resolves into $SRC_REAL" ;; esac
mkdir -p "$DST/src/libero"
DST=$(cd "$DST" && pwd)
for p in "$DST/src" "$DST/src/libero"; do
  case "$(realpath "$p")/" in "$SRC_REAL"/*) die "$p resolves into $SRC_REAL" ;; esac
done
for d in "$SRC"/*; do
  n=$(basename "$d")
  [ "$n" = libero ] && continue
  [ -e "$DST/src/libero/$n" ] || ln -s "$d" "$DST/src/libero/$n"
done
# bench_version stamp (catalog.bench_version reads <root>/../../SC_VERSION)
[ -e "$DST/src/SC_VERSION" ] || ln -s "$SRC_VENVS/libero_pro_src/SC_VERSION" "$DST/src/SC_VERSION"
LIB=$DST/src/libero/libero
"$PY" "$HERE/../render_fix.py" mirror "$SRC/libero" "$LIB"
# apply writes into $LIB/assets: it must be our copy (a reused DST could hold a symlink to the original, and
# `mirror` returns early on an existing path). Real directories, resolving outside the original install,
# no symlinked directory inside assets (files may be symlinks: apply replaces them, never writes through).
for p in "$DST" "$DST/src" "$DST/src/libero" "$LIB" "$LIB/assets"; do
  [ -d "$p" ] && [ ! -L "$p" ] || die "$p is not a real directory"
done
for p in "$LIB" "$LIB/assets"; do
  case "$(realpath "$p")/" in "$SRC_REAL"/*) die "$p resolves into the original install ($SRC_REAL)" ;; esac
done
[ -z "$(find "$LIB/assets" -type l -xtype d -print -quit)" ] || die "symlinked directory inside $LIB/assets"
"$PY" "$HERE/../render_fix.py" apply --assets "$LIB/assets"
cat > "$DST/python" <<EOF
#!/usr/bin/env bash
# LIBERO-PRO venv python with the render-fixed package copy first on the path (fixed_mirror.sh)
export PYTHONPATH="$DST/src\${PYTHONPATH:+:\$PYTHONPATH}"
exec "$PY" "\$@"
EOF
chmod +x "$DST/python"
"$DST/python" -c "import libero.libero as m, sys; print('libero.libero from', m.__file__)"
"$PY" "$HERE/../render_fix.py" status --assets "$DST/src/libero/libero/assets"
