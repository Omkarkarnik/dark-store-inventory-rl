#!/usr/bin/env bash
# Steps 4-5: statistics, tables and figures, and the demo animation.
# About 5 minutes. Run after ./run_experiments.sh has finished.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/04_analyze.py
python scripts/05_demo.py
echo "Done. See results/."
