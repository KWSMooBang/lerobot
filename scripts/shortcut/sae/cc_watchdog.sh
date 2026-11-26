#!/usr/bin/env bash
# T034 8 GPU-h cap: used = finished jobs in gpu_time.tsv + elapsed time of the running GPU job (python cc_train.py /
# collect.py of this task). At the cap the running python job gets SIGTERM (not its run_gpu.sh wrapper, which then
# still writes the ledger row) and a marker file is written; any later job is terminated too.
LEDGER=/outputs/VLA-shortcut-exps/sae/crosscoder/gpu_time.tsv
CAP=${CAP_S:-28800}
MARK=/outputs/VLA-shortcut-exps/sae/crosscoder/logs/CAP_REACHED
PAT="sae/cc_train.py|collect.py --model pi05lib --layers emb"
job_pids() {
  for p in $(pgrep -f "$PAT"); do
    case "$(ps -o comm= -p "$p" 2>/dev/null)" in python*) echo "$p" ;; esac
  done
}
while true; do
  done_s=$(awk -F'\t' '{s+=$4} END {print s+0}' "$LEDGER")
  pids=$(job_pids)
  run_s=0
  for p in $pids; do
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$e" ] && [ "$e" -gt "$run_s" ] && run_s=$e
  done
  used=$((done_s + run_s))
  if [ "$used" -ge "$CAP" ] && [ -n "$pids" ]; then
    echo "$(date '+%F %T') cap reached: used ${used}s, terminating $pids" | tee -a "$MARK"
    kill -TERM $pids
  fi
  sleep 30
done
