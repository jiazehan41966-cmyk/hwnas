#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jiajia/experiments/sonar_phase5a_20260920_v1
PYTHON=/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python
cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u tools/phase5a_quantization/run.py --skip-export
