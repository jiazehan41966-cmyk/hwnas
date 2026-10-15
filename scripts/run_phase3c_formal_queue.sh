#!/usr/bin/env bash
set -euo pipefail

ROOT=${PHASE3C_ROOT:-/home/jiajia/experiments/sonar_phase3c_20260918_v1}
PYTHON=${PHASE3C_PYTHON:-/home/jiajia/experiments/zcu104_full_campaign_v1/.venv/bin/python}
export HWNAS_SOURCE_GIT_COMMIT=${HWNAS_SOURCE_GIT_COMMIT:-8630f124735fec7561c562e7ac8436316c6d38e9}
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p logs/phase3c_formal results/transfer

run_job() {
  local gpu=$1 dataset=$2 model=$3 seed=$4 runner
  if [[ $dataset == UATD ]]; then
    runner=scripts/run_uatd_architecture_transfer.py
  else
    runner=scripts/run_sctd_architecture_transfer.py
  fi
  echo "START dataset=$dataset model=$model seed=$seed gpu=$gpu time=$(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$runner" \
    --mode formal --device cuda --models "$model" --seeds "$seed" \
    --resume --defer-aggregate \
    > "logs/phase3c_formal/${dataset}_${model}_seed${seed}.log" 2>&1
  echo "DONE dataset=$dataset model=$model seed=$seed gpu=$gpu time=$(date --iso-8601=seconds)"
}

worker0() {
  run_job 0 UATD baseline 0
  run_job 0 UATD baseline 1
}
worker1() {
  run_job 1 UATD baseline 2
  run_job 1 UATD nas 0
}
worker2() {
  run_job 2 UATD nas 1
  run_job 2 SCTD baseline 0
  run_job 2 SCTD baseline 1
  run_job 2 SCTD baseline 2
}
worker3() {
  run_job 3 UATD nas 2
  run_job 3 SCTD nas 0
  run_job 3 SCTD nas 1
  run_job 3 SCTD nas 2
}

worker0 > logs/phase3c_formal/worker0.log 2>&1 & p0=$!
worker1 > logs/phase3c_formal/worker1.log 2>&1 & p1=$!
worker2 > logs/phase3c_formal/worker2.log 2>&1 & p2=$!
worker3 > logs/phase3c_formal/worker3.log 2>&1 & p3=$!
printf '%s\n' "$p0" "$p1" "$p2" "$p3" > logs/phase3c_formal/worker_pids.txt

status=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  printf '{"status":"FAILED","test_evaluated":false}\n' > results/transfer/PHASE3C_QUEUE_STATUS.json
  exit 1
fi

"$PYTHON" scripts/finalize_phase3c_transfer.py --mode freeze
CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/finalize_phase3c_transfer.py --mode test --device cuda
printf '{"status":"COMPLETE","formal_runs":"12/12","test_evaluated":true}\n' > results/transfer/PHASE3C_QUEUE_STATUS.json
