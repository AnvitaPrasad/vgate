#!/usr/bin/env bash
# One-off: upload processed tables, frozen splits and images as the private dataset `<user>/vgate-data`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KUSER="$(cat "$ROOT/.kaggle-cfg/username")"
STAGE="$ROOT/kaggle/_dataset_vgate-data"
rm -rf "$STAGE" && mkdir -p "$STAGE/images"
cp -r "$ROOT/data/processed" "$ROOT/data/splits" "$STAGE/"
cp -r "$ROOT/data/raw/evqa_images/evqa_inference_images" "$STAGE/images/evqa"
cp -r "$ROOT/data/raw/infoseek_images/val" "$STAGE/images/infoseek"
cat > "$STAGE/dataset-metadata.json" <<META
{"title": "vgate-data", "id": "$KUSER/vgate-data", "licenses": [{"name": "other"}]}
META
"$ROOT/scripts/kaggle.sh" datasets create -p "$STAGE" -r zip
