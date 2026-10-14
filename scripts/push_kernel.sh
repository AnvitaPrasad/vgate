#!/usr/bin/env bash
# Sync the code dataset, wait for it to publish, bake its build stamp into the job's run.py, then push.
# Usage: scripts/push_kernel.sh kaggle/<job> [--no-sync]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUSER="$(cat "$ROOT/.kaggle-cfg/username")"
DIR="$1"; SYNC="${2:-}"
if [[ "$SYNC" != "--no-sync" ]]; then
  "$ROOT/scripts/sync_kaggle_dataset.sh" "auto-sync before $(basename "$DIR")" >/dev/null
  for _ in $(seq 1 40); do
    s="$("$ROOT/scripts/kaggle.sh" datasets status "$KUSER/vgate-code" 2>&1 | tail -1)"
    [[ "$s" == *ready* ]] && break
    sleep 15
  done
fi
STAMP="$(python3 -c "import re,pathlib;print(re.search(r'BUILD = \"(\w+)\"', pathlib.Path('$ROOT/src/vgate/_build.py').read_text()).group(1))")"
python3 - "$DIR" "$STAMP" "$KUSER" <<'PY'
import json, pathlib, re, sys
d, stamp, user = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p = d / "run.py"; s = p.read_text()
s = re.sub(r'EXPECTED = "[^"]*"', f'EXPECTED = "{stamp}"', s); p.write_text(s)
m = json.loads((d / "kernel-metadata.json").read_text())
m["id"] = f"{user}/{d.name}"
m["dataset_sources"] = [f"{user}/{x.split('/')[-1]}" for x in m.get("dataset_sources", [])]
(d / "kernel-metadata.json").write_text(json.dumps(m, indent=1))
print("stamp", stamp, "->", m["id"])
PY
"$ROOT/scripts/kaggle.sh" kernels push -p "$DIR"
