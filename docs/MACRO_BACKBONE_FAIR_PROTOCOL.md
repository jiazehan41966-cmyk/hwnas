# Macro Backbone Fair Protocol

This document freezes the historical 2026-04-17 fair rerun protocol for the
macro backbone comparison on NKSID + AV7K325. It is retained for artifact
reproduction, not as the current physical-board or generalization protocol.

The explicit `LUT<=50950` value below used the XC7K325T slice count as a LUT
limit. The corrected physical capacity is `203800` LUTs. The fold-0 split also
lacks acquisition-group metadata. New experiments must start from the current
semantic-safe config and first-principles audit rather than silently editing
this frozen protocol.

Canonical config:

- [configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml](../configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml)

## Fixed Protocol Table

| Item | Fair setting | Notes |
| --- | --- | --- |
| Weight initialization | All candidates train from scratch | No pretrained weights in the main table |
| Candidate set | `MobileNetV2`, `FBNet-A`, `ShuffleNetV2`, `EfficientNet-B0` | One candidate per macro family; `SimpleCNN` is excluded from the main fair table |
| Dataset | `NKSID` | Same dataset for every backbone |
| Split | `fold=0`, `use_kfold=true`, `split_seed=42` | Same train/val partition for every backbone |
| Input shape | `1 x 224 x 224` | Same grayscale input for every backbone |
| Number of classes | `8` | Fixed sonar class count |
| Data augmentation | `resize + horizontal_flip + vertical_flip + rotation + affine + color_jitter + speckle_noise + normalize` | Implemented by the shared NKSID pipeline in [dataset.py](../src/hwnas_fpga/data/dataset.py) |
| Optimizer | `AdamW` | Shared optimizer for all candidates |
| Learning rate | `0.001` | No per-backbone tuning in the fair rerun |
| Weight decay | `1e-4` | Shared regularization |
| LR schedule | `5-epoch linear warmup + cosine decay` | Implemented in [run_backbone_baseline.py](../run_backbone_baseline.py) |
| Max epochs | `240` | Same cap for every backbone |
| Early stopping | `patience=20` on `val_macro_f1` | Same stop rule for every backbone |
| Batch size | `64` | Same batch size for every backbone |
| Device policy | Same machine / same precision mode | Keep execution environment fixed across candidates |
| Validation metric | `macro_f1` | Primary selection metric |
| Auxiliary metrics | `top1`, `top5`, `weighted_f1` | Reported, but not used as the main rank metric |
| Hardware target | `alinx_av7k325 @ 200 MHz` | Same FPGA target for all candidates |
| Quantization assumption | `INT8` | Same FPGA estimator assumption |
| Hardware constraints | `DSP<=840`, `BRAM<=445`, `LUT<=50950`, `Power<=12W` | Shared board limits |
| Main ranking rule | `feasible > macro_f1 > top1 > fpga_latency_ms` | The latency tie-break is now configurable and set to FPGA latency |
| Search anchor rule | Best model within `mobilenet_v2/fbnet_a/shufflenet_v2` families | Prevents the accuracy-heavy backbone from becoming the search seed |
| Lightweight anchor rule | Minimum `fpga_latency_ms` within `accuracy_anchor - 0.10 macro_f1` | `lightweight_metric` is now actually honored by code |

## Candidate Table

| Arch ID | Display name | Pretrained | Included in fair main table |
| --- | --- | --- | --- |
| `mobilenet_v2_scratch` | `MobileNetV2` | No | Yes |
| `fbnet_a` | `FBNet-A` | No | Yes |
| `shufflenet_v2_scratch` | `ShuffleNetV2` | No | Yes |
| `efficientnet_b0_scratch` | `EfficientNet-B0` | No | Yes |

## Reporting Recommendation

For the main thesis table:

- Report the scratch-only comparison above as the primary macro comparison.
- If you still want to show transfer-learning ceilings, put pretrained results in
  a separate supplementary table and do not use them for anchor selection.

For randomness control:

- Preferred protocol: run `3` seeds such as `42`, `43`, and `44`, then report
  `mean +/- std` for `macro_f1`.
- If compute is tight, start with the single-seed run below, but label it as a
  provisional result rather than the final thesis table.

## Execution Commands

Single-seed fair rerun:

```bash
python run_backbone_baseline.py --config configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml --run-name backbone_nksid_av7k325_fair_scratch_seed42 --seed 42
```

Recommended 3-seed rerun:

```bash
python run_backbone_baseline.py --config configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml --run-name backbone_nksid_av7k325_fair_scratch_seed42 --seed 42
python run_backbone_baseline.py --config configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml --run-name backbone_nksid_av7k325_fair_scratch_seed43 --seed 43
python run_backbone_baseline.py --config configs/backbone_baseline_nksid_av7k325_fair_scratch.yaml --run-name backbone_nksid_av7k325_fair_scratch_seed44 --seed 44
```

## Expected Outputs

Each run should produce:

- `results/<run_name>/results/backbone_summary.json`
- `results/<run_name>/results/backbone_summary.csv`
- `results/<run_name>/results/best_backbone.json`
- `results/<run_name>/results/selected_backbone_pool.json`
- `results/<run_name>/results/experiment_protocol.json`
