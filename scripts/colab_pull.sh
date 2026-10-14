#!/usr/bin/env bash
# Pull a Colab output dir into experiments/runs/<name>. Usage: scripts/colab_pull.sh <session> <remote_dir> <name>
set -euo pipefail
S="$1"; R="$2"; N="$3"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/experiments/runs/$N"
cd ~
echo "import subprocess;print(subprocess.run('cd $R && tar czf /content/_pull_$N.tgz . && ls -la /content/_pull_$N.tgz',shell=True,capture_output=True,text=True).stdout)" \
  | colab exec -s "$S" --timeout 300 2>&1 | grep -v "new version\|update\|silence" || true
colab download -s "$S" "/content/_pull_$N.tgz" "$ROOT/experiments/runs/_pull_$N.tgz" 2>&1 | grep -v "new version\|update\|silence" | tail -1
tar xzf "$ROOT/experiments/runs/_pull_$N.tgz" -C "$ROOT/experiments/runs/$N" && rm "$ROOT/experiments/runs/_pull_$N.tgz"
ls "$ROOT/experiments/runs/$N" | head -40
