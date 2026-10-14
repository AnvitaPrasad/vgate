#!/usr/bin/env bash
# Project-scoped Kaggle CLI. Never uses ~/.kaggle (that is a different project's account).
# Needs .kaggle-cfg/access_token and .kaggle-cfg/username (both git-ignored).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KAGGLE_API_TOKEN="$(cat "$ROOT/.kaggle-cfg/access_token")"
export KAGGLE_CONFIG_DIR="$ROOT/.kaggle-cfg"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo /opt/anaconda3/bin/kaggle)}"  # anaconda bin may be off PATH
exec "$KAGGLE_BIN" "$@"
