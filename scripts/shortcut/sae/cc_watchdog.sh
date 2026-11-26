#!/usr/bin/env bash
# T034 8 GPU-h cap: used = finished jobs in gpu_time.tsv + elapsed time of the running GPU job (cc_train.py /
# collect.py of this worktree). At the cap the running job gets SIGTERM and a marker file is written.
LEDGER=/outputs/VLA-shortcut-exps/sae/crosscoder/gpu_time.tsv
CAP=${CAP_S:-28800}
MARK=/outputs/VLA-shortcut-exps/sae/crosscoder/logs/CAP_REACHED
while true; do
  done_s=$(awk -F'\t' '{s+=$4} END {print s+0}' "$LEDGER")
  pid=$(pgrep -f "sae/cc_train.py|collect.py --model pi05lib --layers emb" | head -n 1)
  run_s=0
  [ -n "$pid" ] && run_s=$(ps -o etimes= -p "$pid" | tr -d ' ')
  used=$((done_s + ${run_s:-0}))
  if [ "$used" -ge "$CAP" ]; then
    echo "$(date '+%F %T') cap reached: used ${used}s, killing pid ${pid:-none}" | tee -a "$MARK"
    [ -n "$pid" ] && pkill -TERM -f "sae/cc_train.py|collect.py --model pi05lib --layers emb"
    sleep 30; continue
  fi
  sleep 30
done
