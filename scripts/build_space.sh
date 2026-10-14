#!/usr/bin/env bash
# Assemble a hosted build of the log view into a folder. Nothing is uploaded.
#   scripts/build_space.sh [output folder]            the Gradio Space (needs a server; default build/space_log)
#   scripts/build_space.sh --static [output folder]   a static page that runs the log view in the browser on Pyodide, no server (default build/space_static)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--static" ]]; then
  OUT="${2:-$ROOT/build/space_static}"
  rm -rf "$OUT"; mkdir -p "$OUT/files/demo" "$OUT/files/src/vgate"
  cp "$ROOT"/demo/space_static/README.md "$ROOT"/LICENSE "$OUT/"
  cp "$ROOT"/demo/vgate_demo.mp4 "$OUT/vgate_demo.mp4"   # the walkthrough video, linked from the page
  cp "$ROOT"/demo/space_static/glue.py "$OUT/files/glue.py"
  cp "$ROOT"/demo/app.py "$ROOT"/demo/sample_log.csv "$ROOT"/demo/sample_log_published.csv "$OUT/files/demo/"
  for f in __init__ _build gate uncertainty evaluation; do cp "$ROOT/src/vgate/$f.py" "$OUT/files/src/vgate/"; done   # what the log view imports
  LIST="$(cd "$OUT/files" && find . -type f | sed 's|^\./||' | sort | python3 -c 'import sys, json; print(json.dumps(sys.stdin.read().split()))')"
  python3 - "$ROOT/demo/space_static/index.html" "$OUT/index.html" "$LIST" <<'PY'
import pathlib, sys
src, dst, files = sys.argv[1:4]
pathlib.Path(dst).write_text(pathlib.Path(src).read_text().replace("__FILES__", files))
PY
else
  OUT="${1:-$ROOT/build/space_log}"
  rm -rf "$OUT"; mkdir -p "$OUT/demo" "$OUT/src/vgate"
  cp "$ROOT"/demo/space_log/{README.md,requirements.txt,app.py} "$OUT/"
  cp "$ROOT"/demo/app.py "$ROOT"/demo/sample_log.csv "$ROOT"/demo/sample_log_published.csv "$OUT/demo/"
  cp "$ROOT"/src/vgate/*.py "$OUT/src/vgate/"      # the library modules only: no fitted gates, no data
  cp "$ROOT"/LICENSE "$OUT/"
fi
du -sh "$OUT"; find "$OUT" -type f | sed "s|$OUT/||" | sort
