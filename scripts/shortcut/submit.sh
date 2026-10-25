#!/usr/bin/env bash
# Run / submit shortcut evals, one run per (checkpoint, suite, condition, repeat).
#
#   scripts/shortcut/submit.sh [--direct|--launch] [--dry-run] --ckpt pi05lib,pi05base \
#       --suites goal,spat --conds C0,C1,C4 [--repeats 0,1,2] [--episodes 50] [--tasks 0-9] \
#       [--hours 12] [-- <extra run_eval.py args>]
#
# run_id: sc-<ckpt tag>-<suite short: spat|obj|goal|l10>-<cond>[-r<k>]   (e.g. sc-pi05lib-goal-C4, sc-pi05lib-goal-C0-r1)
#   --repeats applies to C0 only (other conditions always repeat 0); "-r<k>" is added for k > 0.
#
# Modes
#   --direct (default)  run sequentially here with CUDA_VISIBLE_DEVICES=0 (GPU 0 only). Writes the same
#                       layout as /harness/launch.sh: /outputs/<project>/runs/<run_id>/{git_sha,git_dirty,
#                       patch.diff,cmd.txt,meta.txt,STATUS,stdout.log,...} and one results.tsv line per run.
#                       An existing run dir is resumed (run_eval.py skips finished episodes).
#   --launch            hand each run to /harness/launch.sh (FIFO queue, picks any free GPU, returns at once).
#   --dry-run           print the run_ids and commands only.
set -euo pipefail

MODE=direct; DRY=0; CKPTS=pi05lib; SUITES=goal; CONDS=C0; REPEATS=0; EPISODES=50; TASKS=0-9; HOURS=12
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --direct) MODE=direct ;;
    --launch) MODE=launch ;;
    --dry-run) DRY=1 ;;
    --ckpt) CKPTS=$2; shift ;;
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

REPO=$(git rev-parse --show-toplevel)
GCD=$(git rev-parse --git-common-dir)
PROJECT=$(basename "$(dirname "$(cd "$GCD" && pwd)")")
OUTPUTS_DIR=/outputs/$PROJECT

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

record_result() {  # same columns as launch.sh: run_id status git_sha env gpu_h ended_at metrics
  local id=$1 st=$2 out=$3 start=$4 end metrics
  end=$(date +%s); date -Iseconds > "$out/ended_at"
  metrics=$(uv run --project "$REPO" python -c '
import json, sys
eps = [json.loads(l) for l in open(sys.argv[1])]
n = len(eps); so = sum(e["success_orig"] for e in eps); dual = [e for e in eps if e.get("success_new") is not None]
s = f"n={n};sr_orig={so / n:.3f}" if n else "n=0"
if dual: s += f";sr_new={sum(e[\"success_new\"] for e in dual) / len(dual):.3f}"
print(s)' "$out/episodes.jsonl" 2>/dev/null || echo -)
  (
    flock 9
    [ -s "$OUTPUTS_DIR/results.tsv" ] || printf 'run_id\tstatus\tgit_sha\tenv\tgpu_h\tended_at\tmetrics\n' >> "$OUTPUTS_DIR/results.tsv"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$st" "$(cut -c1-12 "$out/git_sha")" "direct-gpu0" \
      "$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.2f", (e - s) / 3600 }')" "$(cat "$out/ended_at")" "$metrics" \
      >> "$OUTPUTS_DIR/results.tsv"
  ) 9>"$OUTPUTS_DIR/.results.lock"
}

IFS=, read -r -a CK_A <<< "$CKPTS"; IFS=, read -r -a SU_A <<< "$SUITES"
IFS=, read -r -a CO_A <<< "$CONDS"; IFS=, read -r -a RE_A <<< "$REPEATS"
RUN_IDS=()
for tag in "${CK_A[@]}"; do
  ckpt=$(ckpt_id "$tag")
  for s in "${SU_A[@]}"; do
    suite=$(suite_name "$s")
    for cond in "${CO_A[@]}"; do
      reps=(0); [ "$cond" = C0 ] && reps=("${RE_A[@]}")
      for k in "${reps[@]}"; do
        id="sc-$tag-$s-$cond"; [ "$k" -gt 0 ] && id="$id-r$k"
        out="$OUTPUTS_DIR/runs/$id"
        cmd=(uv run --project "$REPO" python "$REPO/scripts/shortcut/run_eval.py" --policy pi05 --checkpoint "$ckpt"
             --model-tag "$tag" --suite "$suite" --condition "$cond"
             --instr-set "$REPO/scripts/shortcut/instructions/$suite.json" --tasks "$TASKS" --episodes "$EPISODES"
             --repeat-idx "$k" --seed 7 ${EXTRA[@]+"${EXTRA[@]}"})
        RUN_IDS+=("$id")
        if [ "$DRY" = 1 ]; then
          echo "[$MODE] $id: ${cmd[*]} --out $out"
          continue
        fi
        if [ "$MODE" = launch ]; then
          # launch.sh exports RUN_DIR (= runs/<run_id>) to the command
          RUN_HOURS=$HOURS /harness/launch.sh "$id" 1 bash -c "$(printf '%q ' "${cmd[@]}") --out \"\$RUN_DIR\""
          continue
        fi
        mkdir -p "$out"
        git -C "$REPO" rev-parse HEAD > "$out/git_sha"
        git -C "$REPO" status --porcelain > "$out/git_dirty" || true
        git -C "$REPO" diff HEAD --binary > "$out/patch.diff" || true
        printf '%q ' "${cmd[@]}" --out "$out" > "$out/cmd.txt"
        printf 'env=direct-gpu0\nproject=%s\ncwd=%s\ngpus_requested=1\nmode=direct\n' "$PROJECT" "$PWD" > "$out/meta.txt"
        echo running > "$out/STATUS"
        start=$(date +%s)
        echo "[direct] $id → $out"
        if CUDA_VISIBLE_DEVICES=0 RUN_ID=$id RUN_DIR=$out timeout --signal=TERM "${HOURS}h" \
             "${cmd[@]}" --out "$out" >> "$out/stdout.log" 2>&1; then
          st=done
        else
          st=failed
        fi
        echo "$st" > "$out/STATUS"
        record_result "$id" "$st" "$out" "$start"
        echo "[direct] $id: $st"
      done
    done
  done
done
echo "runs=${RUN_IDS[*]}"
