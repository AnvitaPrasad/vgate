#!/usr/bin/env bash
# Drive the A2000 over Tailscale SSH. Host/user in .kaggle-cfg/../.a2000 (git-ignored): A2000=user@host
#   scripts/a2000.sh setup                 # venv in ~/vgate (torch matched to driver per GPU_SETUP_RECIPE), once
#   scripts/a2000.sh sync                  # rsync src/, scripts/, data/{processed,splits}, dev images
#   scripts/a2000.sh run <name> "<cmd>"    # run <cmd> in ~/vgate inside tmux session <name>, log to runs/<name>.log
#   scripts/a2000.sh tail <name> [n]       # last n log lines
#   scripts/a2000.sh pull <name>           # rsync ~/vgate/runs/<name>/ -> experiments/runs/<name>/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A2000="$(cat "$ROOT/.a2000")"
R='~/vgate'
case "$1" in
  setup)
    ssh "$A2000" "mkdir -p $R/runs && cd $R && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv && \
      (command -v python3.10 || command -v python3) && python3 -m venv .venv && . .venv/bin/activate && pip install -q -U pip && \
      pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
      pip install -q 'transformers==5.16.1' accelerate pillow numpy pandas pyarrow scikit-learn rank_bm25 && \
      python -c 'import torch,transformers;print(torch.__version__, torch.cuda.is_available(), transformers.__version__)'" ;;
  sync)
    rsync -az --delete "$ROOT/src/" "$A2000:$R/src/"
    rsync -az "$ROOT/scripts/" "$A2000:$R/scripts/"
    rsync -az "$ROOT/data/processed" "$ROOT/data/splits" "$A2000:$R/data/"
    rsync -az "$ROOT/data/raw/evqa_images/evqa_inference_images/" "$A2000:$R/data/images/evqa/"
    rsync -az "$ROOT/data/raw/infoseek_images/val/" "$A2000:$R/data/images/infoseek/" ;;
  run)
    ssh "$A2000" "tmux new-session -d -s '$2' \"cd $R && . .venv/bin/activate && PYTHONPATH=src $3 > runs/$2.log 2>&1; echo EXIT=\\\$? >> runs/$2.log\"" && echo "started tmux session $2" ;;
  tail)
    ssh "$A2000" "tail -n ${3:-20} $R/runs/$2.log" ;;
  pull)
    mkdir -p "$ROOT/experiments/runs/$2" && rsync -az "$A2000:$R/runs/$2/" "$ROOT/experiments/runs/$2/" ;;
  *) echo "unknown command $1"; exit 1 ;;
esac
