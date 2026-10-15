#!/usr/bin/env bash
set -euo pipefail

ROOT=${PHASE3D_ROOT:-${PHASE3C_ROOT:-/home/jiajia/experiments/sonar_phase3c_20260918_v1}}
PYTHON=${PHASE3D_PYTHON:-${PHASE3C_PYTHON:-/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python}}
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p logs/phase3d_formal results/transfer/phase3d reports/phase3d_structural_decomposition

run_job() {
  local gpu=$1 dataset=$2 seed=$3
  echo "START dataset=$dataset model=B2 seed=$seed gpu=$gpu time=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" scripts/run_phase3d_structural_decomposition.py \
    --mode formal --device cuda --datasets "$dataset" --seeds "$seed" \
    --resume \
    > "logs/phase3d_formal/${dataset}_B2_seed${seed}.log" 2>&1
  echo "DONE dataset=$dataset model=B2 seed=$seed gpu=$gpu time=$(date --iso-8601=seconds)"
}

worker0() { run_job 0 uatd_v3 0; }
worker1() { run_job 1 uatd_v3 1; }
worker2() { run_job 2 uatd_v3 2; }
worker3() {
  run_job 3 classic_sctd_1_0 0
  run_job 3 classic_sctd_1_0 1
  run_job 3 classic_sctd_1_0 2
}

"$PYTHON" scripts/run_phase3d_structural_decomposition.py --mode preflight
worker0 > logs/phase3d_formal/worker0.log 2>&1 & p0=$!
worker1 > logs/phase3d_formal/worker1.log 2>&1 & p1=$!
worker2 > logs/phase3d_formal/worker2.log 2>&1 & p2=$!
worker3 > logs/phase3d_formal/worker3.log 2>&1 & p3=$!
printf '%s\n' "$p0" "$p1" "$p2" "$p3" > logs/phase3d_formal/worker_pids.txt

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  printf '{"status":"FAILED","b2_formal_runs":"incomplete","phase3d_test_accessed":false}\n' \
    > results/transfer/phase3d/PHASE3D_QUEUE_STATUS.json
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_phase3d_structural_decomposition.py --mode evaluate-valid --device cuda
"$PYTHON" scripts/run_phase3d_structural_decomposition.py --mode analyze
printf '{"status":"COMPLETE","b2_formal_runs":"6/6","phase3d_test_accessed":false}\n' \
  > results/transfer/phase3d/PHASE3D_QUEUE_STATUS.json
