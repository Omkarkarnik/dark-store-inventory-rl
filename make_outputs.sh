#!/usr/bin/env bash
# Steps 4-6: statistics, tables and figures; the demo animation; the slide deck and narration script.
# About 5 minutes. Run after ./run_experiments.sh has finished.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/04_analyze.py
python scripts/05_demo.py
python scripts/06_make_presentation.py
echo "Done. See results/ and presentation/."
