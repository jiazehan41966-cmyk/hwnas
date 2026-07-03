#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/hwnas-codex-check
python run_operator_ablation.py \
  --config configs/operator_ablation_nksid_av7k325.yaml \
  --device cuda \
  --selected-backbone-pool results/backbone_nksid_av7k325_fair_scratch_seed42_gpu_20260417/results/selected_backbone_pool.json \
  --pool-role search_anchor \
  --run-name operator_ablation_nksid_av7k325_search_anchor_singleops_ep120_gpu_20260418
