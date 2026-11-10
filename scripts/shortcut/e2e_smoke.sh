#!/usr/bin/env bash
# End-to-end smoke of the testbed (T019): train a small policy → register → one battery over all three
# benchmarks with that checkpoint → sc-report → PASS / FAIL.
#
#   scripts/shortcut/e2e_smoke.sh --out-root DIR [--gpu 0] [--policy sc_template|smolvla] [--steps N]
#                                 [--run-id ID] [--mock | --cpu] [--allow-skip]
#
#   (a) sc-train --preset <policy>_libero_all --steps N --then-bench e2e_smoke   (short run, registered as ID)
#   (b) bench/batteries/e2e_smoke.yaml: ours C0·C4 + LIBERO-Plus language·layout + LIBERO-PRO task·position,
#       libero_goal task 1, 1–2 episodes per variant, run as bench run e2e-bench-<ID>
#   (c) policy_server.py --check on the new checkpoint (load + one inference, CPU), sc-report over the bench run,
#       and a check that every job ran: PASS (exit 0) / FAIL (exit 1)
#
# Modes: default = GPU --gpu (≈ 0.15 GPU-h: ~5 min training incl. data loading + ~4 min bench, sc-bench --dry-run);
#        --cpu  = training and the real envs on CPU (mesa EGL renderer; measured ~1 min training + 16 min bench);
#        --mock = training on CPU, bench with the mock env/policy backend (CPU, ~2 min; plumbing only).
# Needs the LIBERO-Plus and LIBERO-PRO installs (setup_server.sh --with-plus --with-pro, or $LIBERO_PLUS_VENV,
# $LIBERO_PRO_ROOT, $LIBERO_PRO_PYTHON); a missing one is a FAIL unless --allow-skip.
set -euo pipefail

OUT=""; GPU=0; POLICY=sc_template; STEPS=""; RID=""; MODE=gpu; ALLOW_SKIP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out-root) OUT=$2; shift ;;
    --gpu) GPU=$2; shift ;;
    --policy) POLICY=$2; shift ;;
    --steps) STEPS=$2; shift ;;
    --run-id) RID=$2; shift ;;
    --mock) MODE=mock ;;
    --cpu) MODE=cpu ;;
    --allow-skip) ALLOW_SKIP=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$OUT" ] || { echo "--out-root is required" >&2; exit 2; }
case "$POLICY" in sc_template|smolvla) ;; *) echo "--policy: sc_template | smolvla" >&2; exit 2 ;; esac
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SC="$REPO/scripts/shortcut"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
RID=${RID:-e2e-$POLICY-$MODE-$(date +%Y%m%d-%H%M%S)}
BENCH_RID=e2e-bench-$RID
# every run starts clean: an earlier run's outputs must never be what the verdict reads
for d in "$OUT/runs/$RID" "$OUT/runs/$BENCH_RID" "$OUT/reports/$BENCH_RID"; do
  if [ -e "$d" ]; then echo "[e2e] $d exists: use a new --run-id (default: a timestamped one)" >&2; exit 2; fi
done
if [ -f "$OUT/ckpt_registry.tsv" ] && cut -f1 "$OUT/ckpt_registry.tsv" | grep -qx "$RID"; then
  echo "[e2e] ckpt id $RID already in $OUT/ckpt_registry.tsv: use a new --run-id" >&2; exit 2
fi
if [ -z "$STEPS" ]; then
  if [ "$MODE" = gpu ]; then STEPS=300; else STEPS=4; fi
fi
note() { echo "[e2e] $*"; }

train=(--preset "${POLICY}_libero_all" --run-id "$RID" --steps "$STEPS" --batch 8 --effective-batch 8
       --wandb off --out-root "$OUT" --then-bench e2e_smoke)
lerobot=(--save_freq="$STEPS" --log_freq=1)
case "$MODE" in
  gpu)
    train+=(--gpus "$GPU" --bench-args "--run-id $BENCH_RID --gpus $GPU")
    lerobot+=(--num_workers=4) ;;
  cpu)
    train+=(--cpu --bench-args "--run-id $BENCH_RID --device cpu")
    lerobot+=(--num_workers=0) ;;
  mock)
    train+=(--cpu --bench-args "--run-id $BENCH_RID --backend mock")
    lerobot+=(--num_workers=0) ;;
esac
if [ "$MODE" != gpu ]; then
  export CUDA_VISIBLE_DEVICES="" SC_GPU_LOCK_DIR=/nonexistent  # no GPU, no GPU lock
fi

note "(a)+(b) train $RID ($POLICY, $STEPS updates, $MODE) then sc-bench e2e_smoke as $BENCH_RID"
RC_TRAIN=0
"$SC/train/sc-train" "${train[@]}" -- "${lerobot[@]}" || RC_TRAIN=$?
[ "$RC_TRAIN" = 0 ] || note "sc-train (+ --then-bench sc-bench) exit $RC_TRAIN"

note "(c) checkpoint load check (CPU) + sc-report"
CHECK=$(uv run --project "$REPO" --extra shortcut-bench python "$SC/serve/policy_server.py" --check --ckpt "$RID" \
          --registry "$OUT/ckpt_registry.tsv" --device cpu 2>/dev/null | tail -1) || CHECK='{"ok": false, "exit": "nonzero"}'
note "load check: $CHECK"
RC_REPORT=0
"$SC/bench/sc-report" --runs "$OUT/runs/$BENCH_RID" --out "$OUT/reports/$BENCH_RID" --videos 0 || RC_REPORT=$?

uv run --project "$REPO" python - "$OUT" "$RID" "$BENCH_RID" "$CHECK" "$ALLOW_SKIP" "$RC_TRAIN" "$RC_REPORT" <<'EOF'
import json
import sys
from pathlib import Path

out, rid, brid, check, allow_skip = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
rc_train, rc_report = int(sys.argv[6]), int(sys.argv[7])
expected = ["ours-goal-C0", "ours-goal-C4", "plus-goal-language", "plus-goal-layout", "pro-goal-task", "pro-goal-position"]
rows, ok = [], True


def row(what, status, note=""):
    global ok
    rows.append((what, status, note))
    ok &= status in ("PASS", "SKIP")


train = out / "runs" / rid / "run_meta.json"
meta = json.loads(train.read_text()) if train.exists() else {}
row("sc-train exit code (incl. --then-bench)", "PASS" if rc_train == 0 else "FAIL", f"exit {rc_train}")
row("train + register", "PASS" if meta.get("status") == "done" and meta.get("ckpt_id") else "FAIL",
    f"step {meta.get('final_step')}, ckpt_id {meta.get('ckpt_id')}, loss {meta.get('loss_first')} -> {meta.get('loss_last')}")
try:
    c = json.loads(check)
except ValueError:
    c = {"ok": False}
row("checkpoint load (policy server, CPU)", "PASS" if c.get("ok") else "FAIL",
    f"{c.get('policy_type')} chunk {c.get('chunk_size')} n_action_steps {c.get('n_action_steps')} actions {c.get('actions_shape')}")
bdir = out / "runs" / brid
status = (bdir / "STATUS").read_text().strip() if (bdir / "STATUS").exists() else "missing"
summ = json.loads((bdir / "jobs_summary.json").read_text()) if (bdir / "jobs_summary.json").exists() else {}
tb = out / "runs" / rid / "then_bench.txt"
tb_rc = tb.read_text().rsplit("returncode=", 1)[-1].strip() if tb.exists() else "missing"
row("sc-bench (--then-bench)", "PASS" if status == "done" and tb_rc == "0" else "FAIL",
    f"STATUS {status}, exit {tb_rc}")
for j in expected:
    s = summ.get(j)
    if s is None:
        row(f"  job {j}", "SKIP" if allow_skip else "FAIL", "not planned (missing install? see the sc-bench warnings)")
    else:
        n = int(s.split(";")[0].split("=")[1])
        row(f"  job {j}", "PASS" if n > 0 else "FAIL", s)
rep = out / "reports" / brid / "manifest.json"
row("sc-report", "PASS" if rep.exists() and rc_report == 0 else "FAIL", f"exit {rc_report}, {rep.parent}")
w = max(len(r[0]) for r in rows)
for what, st, note in rows:
    print(f"[e2e] {what:<{w}}  {st:<4}  {note}")
print(f"[e2e] {'PASS' if ok else 'FAIL'}  run {rid}, bench {brid}, out-root {out}")
sys.exit(0 if ok else 1)
EOF
