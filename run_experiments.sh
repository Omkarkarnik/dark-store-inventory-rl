#!/usr/bin/env bash
# Steps 1-3: optimal policies and baselines, learning-rate sweep, and the 60 main training runs.
# About 2 hours on a MacBook Pro M1. Safe to stop and restart: finished runs are skipped.
# Usage:  caffeinate -i ./run_experiments.sh
#         caffeinate -i ./run_experiments.sh --steps 1000000   (fallback if time is short)
set -euo pipefail
cd "$(dirname "$0")"
python scripts/01_baselines.py
python scripts/02_tune.py
python scripts/03_train.py "$@"
echo "Training finished. Next: ./make_outputs.sh"
