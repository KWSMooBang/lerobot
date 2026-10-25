#!/usr/bin/env bash
# Run / submit shortcut evals, one run per (checkpoint, suite, condition, repeat).
#
#   scripts/shortcut/submit.sh [--direct|--launch] [--detach|--no-detach] [--dry-run] [--id-prefix p1] \
#       --ckpt pi05lib,pi05base --suites goal,spat --conds C0,C1,C4 [--repeats 0,1,2] [--episodes 50] \
#       [--tasks 0-9] [--hours 12] [--ckpt-extra 'pi05lib=--video first:3'] [-- <extra run_eval.py args>]
#
# run_id: sc-[<prefix>-]<ckpt tag>-<suite short: spat|obj|goal|l10>-<cond>[-r<k>]
#   (e.g. sc-pi05lib-goal-C4, sc-p1-pi05lib-goal-C0-r1). --repeats applies to C0 only (other conditions
#   always repeat 0); "-r<k>" is added for k > 0.
# --ckpt-extra <tag>=<args>  extra run_eval.py args for one checkpoint only (repeatable, word-split),
#                            appended after the common `-- ...` args.
#
# Modes
#   --direct (default)  run sequentially with CUDA_VISIBLE_DEVICES=0 (GPU 0 only). Writes the same layout as
#                       /harness/launch.sh: /outputs/<project>/runs/<run_id>/{git_sha,git_dirty,patch.diff,cmd.txt,
#                       meta.txt,gpus,STATUS,started_at,ended_at,stdout.log,...} and one results.tsv line per run.
#                       An existing run dir is resumed (run_eval.py skips finished episodes).
#     --detach (default) every run dir is created now with STATUS=queued, then the sequential loop (the
#                       "driver") is started with setsid nohup and this script returns. Driver log:
#                       /outputs/<project>/runs/_drivers/<timestamp>.log. The driver runs from this worktree
#                       (code + .venv), so do not delete the worktree while it is alive.
#     --no-detach       run the driver loop in the foreground.
#     The driver holds flock on runs/<id>/.alive of every unfinished run (check-runs.sh / qstat stale check)
#     and, before each run, takes /queue/locks/<GPU 0 uuid>.lock (the lock /harness/launch.sh acquire()
#     uses) and waits until GPU 0 memory.used < 1000 MiB. Other launch.sh jobs cannot get GPU 0 while
#     our run holds it.
#   --launch            hand each run to /harness/launch.sh (FIFO queue, picks ANY free GPU, returns at once).
#                       Refused under the current GPU-0-only rule unless --allow-any-gpu is also given.
#   --dry-run           print the run_ids and commands only.
set -euo pipefail

ARGS=("$@")
MODE=direct; DETACH=1; DRY=0; ANY_GPU=0; PREFIX=; CKPTS=pi05lib; SUITES=goal; CONDS=C0; REPEATS=0
EPISODES=50; TASKS=0-9; HOURS=12
EXTRA=()
declare -A CKPT_EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --direct) MODE=direct ;;
    --launch) MODE=launch ;;
    --detach) DETACH=1 ;;
    --no-detach) DETACH=0 ;;
    --dry-run) DRY=1 ;;
    --allow-any-gpu) ANY_GPU=1 ;;
    --id-prefix) PREFIX=$2; shift ;;
    --ckpt) CKPTS=$2; shift ;;
    --ckpt-extra) CKPT_EXTRA[${2%%=*}]=${2#*=}; shift ;;
    --suites) SUITES=$2; shift ;;
    --conds) CONDS=$2; shift ;;
    --repeats) REPEATS=$2; shift ;;
    --episodes) EPISODES=$2; shift ;;
    --tasks) TASKS=$2; shift ;;
    --hours) HOURS=$2; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done
[[ -z $PREFIX || $PREFIX =~ ^[A-Za-z0-9._]+$ ]] || { echo "invalid --id-prefix: $PREFIX" >&2; exit 1; }

if [ "$MODE" = launch ] && [ "$ANY_GPU" != 1 ]; then
  echo "WARNING: --launch lets /harness/launch.sh pick any free GPU; GPU 0 cannot be pinned there." >&2
  echo "Current rule is GPU 0 only -> refusing. Use --direct (default), or add --allow-any-gpu once GPUs 1-3 are allowed." >&2
  exit 2
fi

REPO=$(git rev-parse --show-toplevel)
GCD=$(git rev-parse --git-common-dir)
PROJECT=$(basename "$(dirname "$(cd "$GCD" && pwd)")")
OUTPUTS_DIR=/outputs/$PROJECT
SELF="$REPO/scripts/shortcut/submit.sh"
MEM_FREE_MB=1000; POLL_SEC=60

ckpt_id() {
  case "$1" in
    pi05lib) echo lerobot/pi05_libero_finetuned ;;
    pi05base) echo lerobot/pi05_base ;;
    *) echo "unknown ckpt tag: $1 (pi05lib|pi05base)" >&2; exit 1 ;;
  esac
}
suite_name() {
  case "$1" in
    spat) echo libero_spatial ;; obj) echo libero_object ;; goal) echo libero_goal ;; l10) echo libero_10 ;;
    *) echo "unknown suite: $1 (spat|obj|goal|l10)" >&2; exit 1 ;;
  esac
}
for tag in "${!CKPT_EXTRA[@]}"; do ckpt_id "$tag" > /dev/null; done

record_result() {  # same columns as launch.sh: run_id status git_sha env gpu_h ended_at metrics
  local id=$1 st=$2 out=$3 start=$4 end metrics
  end=$(date +%s); date -Iseconds > "$out/ended_at"
  metrics=$(uv run --project "$REPO" python -c '
import json, sys
eps = [json.loads(l) for l in open(sys.argv[1])]
n = len(eps); so = sum(e["success_orig"] for e in eps); dual = [e for e in eps if e.get("success_new") is not None]
s = f"n={n};sr_orig={so / n:.3f}" if n else "n=0"
sn = sum(e["success_new"] for e in dual)
if dual: s += f";sr_new={sn / len(dual):.3f}"
print(s)' "$out/episodes.jsonl" 2>/dev/null || echo -)
  (
    flock 9
    [ -s "$OUTPUTS_DIR/results.tsv" ] || printf 'run_id\tstatus\tgit_sha\tenv\tgpu_h\tended_at\tmetrics\n' >> "$OUTPUTS_DIR/results.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$st" "$(cut -c1-12 "$out/git_sha")" "direct-gpu0" \
      "$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", (e - s) / 3600 }')" "$(cat "$out/ended_at")" "$metrics" \
      >> "$OUTPUTS_DIR/results.tsv"
  ) 9>>"$OUTPUTS_DIR/.results.lock"
}

gpu0() {  # "<uuid> <memory.used MiB>" of GPU index 0
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 == 0 { print $2, $3 }'
}

# ---------- run list (identical in the submitting process and the driver) ----------
IFS=, read -r -a CK_A <<< "$CKPTS"; IFS=, read -r -a SU_A <<< "$SUITES"
IFS=, read -r -a CO_A <<< "$CONDS"; IFS=, read -r -a RE_A <<< "$REPEATS"
RUN_IDS=(); RUN_OUTS=(); RUN_CMDS=()   # RUN_CMDS[i] = printf %q of the command incl. --out
for tag in "${CK_A[@]}"; do
  ckpt=$(ckpt_id "$tag")
  read -r -a cx <<< "${CKPT_EXTRA[$tag]:-}"
  for s in "${SU_A[@]}"; do
    suite=$(suite_name "$s")
    for cond in "${CO_A[@]}"; do
      reps=(0); [ "$cond" = C0 ] && reps=("${RE_A[@]}")
      for k in "${reps[@]}"; do
        id="sc-${PREFIX:+$PREFIX-}$tag-$s-$cond"; [ "$k" -gt 0 ] && id="$id-r$k"
        out="$OUTPUTS_DIR/runs/$id"
        cmd=(uv run --project "$REPO" python "$REPO/scripts/shortcut/run_eval.py" --policy pi05 --checkpoint "$ckpt"
             --model-tag "$tag" --suite "$suite" --condition "$cond"
             --instr-set "$REPO/scripts/shortcut/instructions/$suite.json" --tasks "$TASKS" --episodes "$EPISODES"
             --repeat-idx "$k" --seed 7 ${EXTRA[@]+"${EXTRA[@]}"} ${cx[@]+"${cx[@]}"})
        RUN_IDS+=("$id"); RUN_OUTS+=("$out")
        if [ "$MODE" = launch ]; then
          # launch.sh exports RUN_DIR (= runs/<run_id>) to the command
          RUN_CMDS+=("$(printf '%q ' "${cmd[@]}")--out \"\$RUN_DIR\"")
        else
          RUN_CMDS+=("$(printf '%q ' "${cmd[@]}" --out "$out")")
        fi
      done
    done
  done
done
N=${#RUN_IDS[@]}

# ---------- driver: sequential loop on GPU 0 ----------
driver() {
  local i id out fd st rc start uuid used gfd
  local -a ALIVE=()
  echo "[driver] pid=$$ repo=$REPO runs=$N"
  for i in "${!RUN_IDS[@]}"; do
    exec {fd}>>"${RUN_OUTS[$i]}/.alive"
    flock -n "$fd" || { echo "[driver] ${RUN_IDS[$i]}: .alive is held by another process" >&2; exit 1; }
    ALIVE[i]=$fd
  done
  echo "[driver] holding .alive locks"
  read -r uuid _ <<< "$(gpu0)"
  [ -n "$uuid" ] || { echo "[driver] GPU 0 not found" >&2; exit 1; }
  mkdir -p /queue/locks
  for i in "${!RUN_IDS[@]}"; do
    id=${RUN_IDS[$i]}; out=${RUN_OUTS[$i]}
    echo "[driver] $(date -Is) $id: waiting for GPU 0 lock /queue/locks/$uuid.lock"
    exec {gfd}>"/queue/locks/$uuid.lock"
    flock "$gfd"
    while read -r _ used <<< "$(gpu0)"; [ "${used:-99999}" -ge "$MEM_FREE_MB" ]; do
      echo "[driver] $(date -Is) $id: GPU 0 memory.used=${used} MiB >= $MEM_FREE_MB, waiting ${POLL_SEC}s"
      sleep "$POLL_SEC"
    done
    date -Is > "$out/started_at"
    echo running > "$out/STATUS"
    start=$(date +%s)
    echo "[driver] $(date -Is) $id: running → $out"
    set +e
    (
      for fd in "${ALIVE[@]}"; do [ "$fd" = "${ALIVE[$i]}" ] || exec {fd}>&-; done   # child keeps only its own
      export CUDA_VISIBLE_DEVICES=0 RUN_ID=$id RUN_DIR=$out
      eval "timeout --signal=TERM --kill-after=10m ${HOURS}h ${RUN_CMDS[$i]}"
    ) >> "$out/stdout.log" 2>&1 < /dev/null
    rc=$?
    set -e
    case $rc in 0) st=done ;; 124) st=timeout ;; *) st=failed ;; esac
    echo "$st" > "$out/STATUS"
    record_result "$id" "$st" "$out" "$start"
    exec {gfd}>&-                                    # release GPU 0
    fd=${ALIVE[$i]}; exec {fd}>&-; unset 'ALIVE[i]'
    echo "[driver] $(date -Is) $id: $st (rc=$rc)"
  done
  echo "[driver] $(date -Is) all done"
}

if [ -n "${SC_DRIVER:-}" ]; then
  driver
  exit 0
fi

# ---------- submit ----------
for i in "${!RUN_IDS[@]}"; do
  if [ "$DRY" = 1 ]; then
    echo "[$MODE] ${RUN_IDS[$i]}: ${RUN_CMDS[$i]}"
  elif [ "$MODE" = launch ]; then
    RUN_HOURS=$HOURS /harness/launch.sh "${RUN_IDS[$i]}" 1 bash -c "${RUN_CMDS[$i]}"
  fi
done
if [ "$DRY" = 1 ] || [ "$MODE" = launch ]; then
  echo "runs=${RUN_IDS[*]}"
  exit 0
fi

# direct: refuse if any run is being driven right now, then mark every run queued
for out in "${RUN_OUTS[@]}"; do
  if [ -f "$out/.alive" ] && ! flock -n "$out/.alive" true; then
    echo "$(basename "$out") is already queued/running under a live driver -> refusing" >&2
    exit 1
  fi
done
TS=$(date +%Y%m%d-%H%M%S)
DLOG="$OUTPUTS_DIR/runs/_drivers/$TS.log"
mkdir -p "$OUTPUTS_DIR/runs/_drivers"
for i in "${!RUN_IDS[@]}"; do
  out=${RUN_OUTS[$i]}
  mkdir -p "$out"
  git -C "$REPO" rev-parse HEAD > "$out/git_sha"
  git -C "$REPO" status --porcelain > "$out/git_dirty" || true
  git -C "$REPO" diff HEAD --binary > "$out/patch.diff" || true
  echo "${RUN_CMDS[$i]}" > "$out/cmd.txt"
  printf 'env=direct-gpu0\nproject=%s\ncwd=%s\ngpus_requested=1\nmode=direct\nrun_hours=%s\ndriver_log=%s\n' \
    "$PROJECT" "$PWD" "$HOURS" "$DLOG" > "$out/meta.txt"
  echo 0 > "$out/gpus"
  rm -f "$out/started_at" "$out/ended_at"
  echo queued > "$out/STATUS"
done

if [ "$DETACH" = 0 ]; then
  SC_DRIVER=1 "$SELF" "${ARGS[@]}" 2>&1 | tee -a "$DLOG"
  exit "${PIPESTATUS[0]}"
fi

SC_DRIVER=1 setsid nohup "$SELF" "${ARGS[@]}" >> "$DLOG" 2>&1 < /dev/null &
PID=$!
for _ in $(seq 60); do   # return only once the driver holds the .alive locks (else check-runs sees stale)
  grep -q '^\[driver\] holding .alive locks' "$DLOG" 2>/dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "driver exited early, see $DLOG" >&2; tail -5 "$DLOG" >&2; exit 1; }
  sleep 0.5
done
grep -q '^\[driver\] holding .alive locks' "$DLOG" || { echo "driver did not take the locks in 30 s, see $DLOG" >&2; exit 1; }
for out in "${RUN_OUTS[@]}"; do echo "pid=$PID log=$DLOG" > "$out/driver"; done
echo "driver pid=$PID log=$DLOG worktree=$REPO"
echo "runs=${RUN_IDS[*]}"
