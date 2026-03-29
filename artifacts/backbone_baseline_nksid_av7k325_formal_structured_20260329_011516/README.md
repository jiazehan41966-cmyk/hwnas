# Backbone Baseline Formal Structured Results (backbone_nksid_av7k325_gpu_formal_structured_20260329_011516)

## Source
- Remote run: `backbone_nksid_av7k325_gpu_formal_structured_20260329_011516`
- Dataset: `NKSID`
- Board: `ALINX AV7K325 @ 200MHz`
- Status: completed
- Purpose: rerun the formal backbone baseline with structured result exports enabled

## What This Artifact Adds
- `results/experiment_protocol.json` with complete candidate-level architecture blueprints
- `results/backbones/*.json` including `architecture_summary` and `training_strategy`
- `results/candidates.json` and `results/candidates.csv` for candidate inventory

## Best Backbone
- Winner: `EfficientNet-B0` (`efficientnet_b0`)
- Macro-F1: `0.9856`
- Top-1: `0.9942`
- FPGA latency: `9.7673 ms`
- CPU latency: `200.6965 ms`

## Recommended Anchors
- accuracy_anchor: `efficientnet_b0`
- search_anchor: `shufflenet_v2`
- lightweight_anchor: `shufflenet_v2`

## Ranking
- `efficientnet_b0`: macro_f1=0.9856, top1=0.9942, fpga_latency_ms=9.7673, feasible=true
- `shufflenet_v2`: macro_f1=0.9794, top1=0.9923, fpga_latency_ms=3.2269, feasible=true
- `mobilenet_v2_scratch`: macro_f1=0.9575, top1=0.9750, fpga_latency_ms=8.5121, feasible=true
- `mobilenet_v2_pretrained`: macro_f1=0.9545, top1=0.9788, fpga_latency_ms=8.5121, feasible=true
- `fbnet_like`: macro_f1=0.8573, top1=0.9135, fpga_latency_ms=6.5793, feasible=true
- `simplecnn`: macro_f1=0.8133, top1=0.9154, fpga_latency_ms=7.2679, feasible=true

## Notes
- This artifact exists alongside the earlier `formal_20260329` sync because that older run did not include `experiment_protocol.json`.
- CPU latency on the shared server is noisy; use FPGA latency as the deployment-facing metric.
- Checkpoints were intentionally excluded from the repository sync to avoid committing large binary artifacts.
