# HLS LUT Builder

This directory contains the formal operator-level LUT build flow for HW-NAS.
The current formal configuration is the MobileNetV2-derived kernel set used for
the LUT sampling stage, not a statement that every runtime consumer in the repo
already speaks the same operator vocabulary.

Primary entry points:

- [candidate_kernels.yaml](/E:/1/hwnas/hwnas/hls_lut_builder/configs/candidate_kernels.yaml)
- [LATENCY_SCOPE.md](/E:/1/hwnas/hwnas/hls_lut_builder/LATENCY_SCOPE.md)
- [kernel_specification.md](/E:/1/hwnas/hwnas/hls_lut_builder/docs/kernel_specification.md)

## Core Operator Set

Mandatory kernels in the first formal LUT build:

- `stem_conv_k3_s2`
- `dw_conv_k3`
- `dw_conv_k5`
- `pw_conv`
- `mbconv_e3_k3`
- `mbconv_e3_k5`
- `mbconv_e6_k3`
- `mbconv_e6_k5`
- `skip`
- `global_avg_pool`
- `fc_layer`

Extension kernels are present in the config but disabled by default until the
v1 operator screening confirms that they should enter the LUT:

- `fused_mbconv_e3_k3`
- `fused_mbconv_e6_k3`
- `mixconv`
- `denoise`
- `edge`

For new NAS runs, hardware feasibility alone is not sufficient for operator
admission. The semantic-safe policy is
`configs/operator_manifest_semantic_safe.yaml`; it pauses `denoise`, `edge`,
and `mixconv`. The canonical search config loads this policy explicitly.
Historical manifests remain unchanged for reproducibility.

Explicitly excluded from this round:

- `7x7` and larger kernels
- `SE`
- `Swish/GELU`
- dilated convolution

## Unified Kernel Contract

Every operator kernel uses the same external contract:

- control: `AXI-Lite`
- feature input: packed `AXI-Stream`, width = `in_channels * 8`
- feature output: packed `AXI-Stream`, width = `out_channels * 8`
- weights: `BRAM`
- biases: `BRAM`
- activation / weight type: `ap_int<8>`
- accumulator type: `ap_int<32>`
- target clock: `5 ns` (`200 MHz`)

The packed-stream contract is the formal LUT measurement policy. It has been
validated on the `stem_conv_k3_s2` pilot and is the contract that future
kernel migrations should match.

All templates keep the same baseline pragma policy:

- interface pragmas are identical across kernels
- loop bodies use `#pragma HLS PIPELINE II=1`
- channel-sensitive local buffers use `ARRAY_PARTITION`
- `DATAFLOW` is intentionally not used in this LUT stage

## Representative Shape Coverage

The first formal core matrix is built from MobileNetV2 anchor stages:

1. Stem: `224 -> 112`, `1 -> 32`
2. Stage1: `112`, `32 -> 16`
3. Stage2: `112 -> 56`, `16 -> 24`
4. Stage3: `56 -> 28`, `24 -> 32`
5. Stage4: `28 -> 14`, `32 -> 64`
6. Stage5: `14`, `64 -> 96`
7. Stage6: `14 -> 7`, `96 -> 160`
8. Stage7: `7`, `160 -> 320`
9. Head: `7 -> 1`, `320 -> 1280`

The primitive kernel matrix expands these stage anchors into:

- 13 depthwise representative shapes
- 26 pointwise representative shapes
- 6 MBConv stage shapes for each `(expand_ratio, kernel)` pair
- 7 skip shapes
- 1 global average pool shape
- 1 FC shape

With only the 11 core kernels enabled, the default config generates `86` HLS
cases, which stays inside the target range of roughly `80-100` core samples.

## Board And Part Note

The current HLS tool part remains `xc7k325t-ffg676-2`. This matches the same
chip family as the project board profile, but not the same package as the
formal `ALINX AV7K325 (XC7K325T-2FFG900I)` board definition. Treat it as a
temporary HLS part for operator LUT sampling, not as full board-level closure.

## Current LUT Query Alignment

This config is now the formal HLS-side kernel set. The runtime LUT query path
already normalizes a subset of these names back into the current NAS operator
vocabulary:

- `stem_conv_k3_s2 -> conv`
- `pw_conv -> conv`
- `mbconv_e*_k* -> mbconv`
- `fused_mbconv_e*_k3 -> fused_mbconv`

Standalone `dw_conv_*`, `global_avg_pool`, and `fc_layer` entries are retained
in the LUT manifest for future use, but they are not yet consumed by every
network-level cost path.

## Recommended Flow

Generate projects:

```powershell
python hls_lut_builder/scripts/gen_project.py --config hls_lut_builder/configs/candidate_kernels.yaml --force
```

Check the command list first:

```powershell
python hls_lut_builder/scripts/run_synthesis.py --config hls_lut_builder/configs/candidate_kernels.yaml --dry-run
```

Run synthesis:

```powershell
python hls_lut_builder/scripts/run_synthesis.py --config hls_lut_builder/configs/candidate_kernels.yaml --vitis-hls "F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat"
```

Run synthesis and immediately attach the downstream Vivado OOC check for each
successful or already-synthesized case:

```powershell
python hls_lut_builder/scripts/run_synthesis.py --config hls_lut_builder/configs/candidate_kernels.yaml --vitis-hls "F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat" --downstream-check --vivado "F:\vivado\Vivado\2023.2\bin\vivado.bat"
```

Parse reports:

```powershell
python hls_lut_builder/scripts/parse_reports.py --config hls_lut_builder/configs/candidate_kernels.yaml
```

Run downstream Vivado on all already-synthesized cases:

```powershell
python hls_lut_builder/scripts/run_vivado_downstream.py --config hls_lut_builder/configs/candidate_kernels.yaml --vivado "F:\vivado\Vivado\2023.2\bin\vivado.bat"
```

Parse downstream Vivado reports:

```powershell
python hls_lut_builder/scripts/parse_vivado_downstream.py --config hls_lut_builder/configs/candidate_kernels.yaml
```

Run the one-case-per-operator pilot pipeline end to end:

```powershell
python hls_lut_builder/scripts/run_pilot_pipeline.py --config hls_lut_builder/configs/candidate_kernels.yaml --vitis-hls "F:\vivado\Vitis_HLS\2023.2\bin\vitis_hls.bat" --vivado "F:\vivado\Vivado\2023.2\bin\vivado.bat"
```

Every major script in the flow now also supports `--pilot`, which picks the
first representative case for each enabled operator according to the formal
operator order in the config.
