#!/usr/bin/env bash
# Drive the GPU server from the laptop.
#   scripts/remote.sh check                 # GPU usage, disk, python
#   scripts/remote.sh sync                  # push code + data (not runs/)
#   scripts/remote.sh run <name> "<cmd>"    # run in background on the server, log to logs/<name>.log
#   scripts/remote.sh log <name> [lines]    # tail a log
#   scripts/remote.sh pull                  # fetch runs/ and data/sft back
set -euo pipefail
HOST=${HOST:-user_joaoandrade@10.100.60.21}
DIR=${DIR:-kimi_multiagent}

case "$1" in
  check) ssh "$HOST" "nvidia-smi; df -h ~; python3 --version; ls ~/.local/bin/uv 2>/dev/null || echo 'no uv'" ;;
  sync)  tar czf - --exclude=.venv --exclude=runs --exclude=.git --exclude=__pycache__ \
           src tests scripts pyproject.toml data third_party/MedCalc-Bench/calculator_implementations \
         | ssh "$HOST" "mkdir -p $DIR/logs && tar xzf - -C $DIR" ;;
  run)   ssh "$HOST" "cd $DIR && nohup bash -lc '$3' > logs/$2.log 2>&1 & echo started $2" ;;
  log)   ssh "$HOST" "tail -n ${3:-30} $DIR/logs/$2.log" ;;
  pull)  ssh "$HOST" "cd $DIR && tar czf - runs data/sft 2>/dev/null" | tar xzf - ;;
  *)     echo "unknown command: $1"; exit 1 ;;
esac
