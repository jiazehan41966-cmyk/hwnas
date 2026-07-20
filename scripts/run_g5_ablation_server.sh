#!/usr/bin/env bash
# G5 E1 four-way ablation, server edition (Linux).
#
# Launches the four matched variants as parallel processes, one GPU each
# (round-robin if fewer GPUs than variants; the models are ~20K params and
# use 2-3 GB VRAM, so stacking 2+ per GPU is fine).
#
# Protocol notes:
# - keep batch 8 x accumulation 4 (effective 32): byte-identical recipe to
#   the local G1 baselines; do NOT trade it for a larger batch.
# - each variant is a single invocation so every record in a run dir shares
#   one run fingerprint (claimable). Re-running resumes per (fold, seed).
# - all four variants must run in this same environment; never merge records
#   produced on a different machine into these run dirs.
#
# Usage:
#   GPUS="0,1,2,3" bash scripts/run_g5_ablation_server.sh
#   GPUS="0" bash scripts/run_g5_ablation_server.sh   # single-GPU stacking

set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-python}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
VARIANTS=(mbconv_control denoise edge denoise_edge)
mkdir -p results/protocol

pids=()
for i in "${!VARIANTS[@]}"; do
    v="${VARIANTS[$i]}"
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    name="g5_ablation_${v}"
    log="results/protocol/${name}.launcher.log"
    echo "[server-queue] $(date -Is) starting ${name} on GPU ${gpu}" | tee -a "$log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 nohup "$PY" run_eval_protocol.py \
        --data-dir data/NKSID \
        --output-dir results/protocol \
        --folds 0,1,2,3,4 \
        --seeds 42,43,44 \
        --epochs 150 \
        --batch-size 8 \
        --gradient-accumulation-steps 4 \
        --amp \
        --save-checkpoints \
        --resume \
        --device cuda \
        --candidate-path "configs/ablation/sonar_g5_v1/${v}.candidate.json" \
        --run-name "$name" >> "$log" 2>&1 &
    pids+=($!)
done

echo "[server-queue] launched ${#pids[@]} variants: ${pids[*]}"
echo "[server-queue] waiting; safe to Ctrl-C (runs continue under nohup)"
wait
echo "[server-queue] $(date -Is) all variants finished"
