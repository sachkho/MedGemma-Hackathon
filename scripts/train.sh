#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
pip install -r requirements.txt
python src/train_medgemma.py --config_file_path config.yaml
