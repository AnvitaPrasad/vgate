#!/usr/bin/env bash
# Resilient Colab runner for free-tier T4s (which get preempted). Loops until the job prints <DONE> or MAX_H hours:
#   allocate T4 (retry every 15 min) -> rebuild bundle from base + local run dir (so finished rows are skipped)
#   -> upload in 40 MB parts (md5-checked) -> launch -> pull outputs after each finished condition -> on VM loss, repeat.
# Usage: scripts/colab_autorun.sh <session> <base_bundle_dir> <launch.py> <remote_out> <local_run_name> <DONE> [MAX_H]
set -uo pipefail
S="$1"; BASE="$2"; LAUNCH="$3"; ROUT="$4"; NAME="$5"; DONE="$6"; MAXH="${7:-10}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; LOC="$ROOT/experiments/runs/$NAME"
WORK="$(dirname "$BASE")/autorun_$NAME"; mkdir -p "$WORK" "$LOC"
say() { echo "$(date +%H:%M) $*"; }
alive() { local o; o=$(cd ~ && colab exec -s "$S" --timeout 60 <<<"print('ping')" 2>&1); [[ "$o" == *ping* ]]; }
cx() { (cd ~ && colab exec -s "$S" --timeout "${2:-120}" 2>/dev/null <<<"$1" | grep -v "new version\|update\|silence" | grep -v '^[[:space:]]*$'); }
end=$(( $(date +%s) + MAXH * 3600 ))
while (( $(date +%s) < end )); do
  if [[ "$(cx "print('ping')" 60)" == *ping* ]]; then say "REUSING live session $S"
  elif (cd ~ && colab new --gpu T4 -s "$S" 2>&1 | grep -q "Session READY"); then say "ALLOCATED"
  else say "ALLOC_REFUSED retry in 15m"; sleep 900; continue; fi
  rm -rf "$WORK/b" && mkdir -p "$WORK/b/J" && cp -r "$BASE/." "$WORK/b/" && cp -r "$ROOT/src" "$WORK/b/" && cp "$LOC"/*.jsonl "$LOC"/*.npy "$LOC"/*.json "$WORK/b/J/" 2>/dev/null
  rm -rf "$WORK/b/J1" && mv "$WORK/b/J" "$WORK/b/J1"
  (cd "$WORK/b" && tar czf "$WORK/bundle.tgz" .) && rm -f "$WORK"/part_* && split -b 40m "$WORK/bundle.tgz" "$WORK/part_"
  ok=1; for f in "$WORK"/part_*; do (cd ~ && colab upload -s "$S" "$f" "/content/$(basename "$f")" >/dev/null 2>&1) || ok=0; done
  want=$(md5 -q "$WORK/bundle.tgz")
  got=$(cx "import subprocess;print(subprocess.run('cat /content/part_* > /content/bundle_j1r.tgz && rm /content/part_* && md5sum /content/bundle_j1r.tgz',shell=True,capture_output=True,text=True).stdout)" | grep -oE '[0-9a-f]{32}' | head -1)
  if [[ $ok != 1 || "$got" != "$want" ]]; then say "UPLOAD_FAILED ($got vs $want)"; (cd ~ && colab stop -s "$S" >/dev/null 2>&1); sleep 300; continue; fi
  (cd ~ && colab exec -s "$S" -f "$LAUNCH" --timeout 300 2>&1 | grep -v "new version\|update\|silence" | tail -1) | sed "s/^/$(date +%H:%M) /"
  last=""
  while true; do
    sleep 180
    st=$(cx "import subprocess;print(subprocess.run('grep -E \"to run|=== |$DONE|Traceback|CUDA out\" $ROUT/log.txt | tail -3',shell=True,capture_output=True,text=True).stdout)" 60)
    if [[ -z "$st" ]] && ! alive; then say "VM_LOST"; break; fi
    if [[ -z "$st" ]]; then continue; fi
    cur=$(echo "$st" | tail -1)
    if [[ "$cur" != "$last" ]]; then
      "$ROOT/scripts/colab_pull.sh" "$S" "$ROUT" "$NAME" >/dev/null 2>&1 && say "PULLED after: $cur ($(cat "$LOC"/*.jsonl | wc -l | tr -d ' ') rows)"
      echo "$st" | grep -q "Traceback\|CUDA out" && say "JOB_ERROR: $st"
      last="$cur"
    fi
    if echo "$st" | grep -q "$DONE"; then "$ROOT/scripts/colab_pull.sh" "$S" "$ROUT" "$NAME" >/dev/null 2>&1; say "DONE"; (cd ~ && colab stop -s "$S" >/dev/null 2>&1); exit 0; fi
  done
done
say "GAVE_UP after ${MAXH}h"
