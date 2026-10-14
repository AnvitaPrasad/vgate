#!/usr/bin/env bash
# Tail a background job's log on a Colab session. Usage: scripts/colab_tail.sh <session> <remote_log> [n]
S="$1"; LOG="$2"; N="${3:-15}"
cd ~ && echo "import subprocess;print(subprocess.run('grep -v -iE \"warn|deprecat|^\\\\s*\$\" $LOG | tail -$N; ls \$(dirname $LOG)',shell=True,capture_output=True,text=True).stdout)" \
  | colab exec -s "$S" --timeout 60 2>&1 | grep -v "new version\|update\|silence"
