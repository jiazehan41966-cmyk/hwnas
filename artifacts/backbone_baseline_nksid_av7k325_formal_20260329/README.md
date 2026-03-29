# Backbone Baseline Formal Results (backbone_nksid_av7k325_gpu_formal_20260329_000614)

## Source
- Remote run: `backbone_nksid_av7k325_gpu_formal_20260329_000614`
- Dataset: `NKSID`
- Board: `ALINX AV7K325 @ 200MHz`
- Status: completed

## Best Backbone
- Winner: `EfficientNet-B0` (`efficientnet_b0`)
- Macro-F1: `0.9801`
- Top-1: `0.9904`
- FPGA latency: `9.7673 ms`
- CPU latency: `17.1177 ms`

## Recommended Anchors
- accuracy_anchor: `efficientnet_b0`
- search_anchor: `shufflenet_v2`
- lightweight_anchor: `shufflenet_v2`

## Ranking
- `efficientnet_b0`: macro_f1=0.9801, top1=0.9904, fpga_latency_ms=9.7673, feasible=true
- `shufflenet_v2`: macro_f1=0.9748, top1=0.9885, fpga_latency_ms=3.2269, feasible=true
- `mobilenet_v2_pretrained`: macro_f1=0.9622, top1=0.9827, fpga_latency_ms=8.5121, feasible=true
- `fbnet_like`: macro_f1=0.9511, top1=0.9750, fpga_latency_ms=6.5793, feasible=true
- `mobilenet_v2_scratch`: macro_f1=0.9013, top1=0.9596, fpga_latency_ms=8.5121, feasible=true
- `simplecnn`: macro_f1=0.8133, top1=0.9154, fpga_latency_ms=7.2679, feasible=true

## Notes
- `ShuffleNetV2` is the strongest deployment-oriented backbone in this run: near-best accuracy with the lowest FPGA latency.
- `EfficientNet-B0` is the pure accuracy winner, but it is slower and larger than `ShuffleNetV2`.
- `MobileNetV2 pretrained` clearly outperforms `MobileNetV2 scratch`, indicating pretrained initialization matters on NKSID.
- Checkpoints were intentionally excluded from the repository sync to avoid committing large binary artifacts.
