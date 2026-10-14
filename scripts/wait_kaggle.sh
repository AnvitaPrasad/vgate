#!/usr/bin/env bash
# Block until all given kernels leave RUNNING/QUEUED. Usage: scripts/wait_kaggle.sh slug1 slug2 ...
# An empty/failed status reply counts as BUSY (the API returns nothing on transient errors).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUSER="$(cat "$ROOT/.kaggle-cfg/username")"
misses=0
while true; do
  line=""; busy=0
  for k in "$@"; do
    s=$("$ROOT/scripts/kaggle.sh" kernels status "$KUSER/$k" 2>&1 | grep -oE 'RUNNING|COMPLETE|ERROR|QUEUED|CANCEL[A-Z_]*' | head -1)
    if [[ -z "$s" ]]; then s="NO_REPLY"; busy=1; misses=$((misses+1)); fi
    [[ "$s" == RUNNING || "$s" == QUEUED ]] && busy=1
    line="$line $k=$s"
  done
  echo "$(date +%H:%M)$line"
  [[ $busy == 0 ]] && exit 0
  if (( misses > 40 )); then echo "giving up after 40 empty replies"; exit 2; fi
  sleep 60
done
