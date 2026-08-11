# ZCU104 Stage4 MBConv-K3-E3 DMA shell pilot

## Scope

This pilot preserves the frozen DMA shell commit `47d4945` and integrates only
`stage4__mbconv_k3_e3__32x28x28_to_32x28x28__s1`. It does not alter the search
space, train a network, change operator mathematics, integrate any other core,
run board software, or measure latency or power.

The generated operator source is retained byte-for-byte after LF normalization
except for interface-only edits: its software-visible HLS AXI4-Lite return port
becomes direct `ap_ctrl_hs`, and the weights/biases BRAM ports are constrained
to single-port access so the other physical ports remain available to the two
AXI BRAM Controllers. The wrapper remains the only control register bank.
Weights use a 7,008-byte BRAM master; biases use 224 signed INT32 entries
(896 bytes). PS access remains at `0xA0100000` and `0xA0200000`.
The unchanged parameter-copy loops are grouped into one `load_parameters`
dataflow process solely to satisfy the one-reader scheduling rule of `ram_1p`.

## Frozen target

- Server: `ssh -p 7889 jiajia@10.70.23.121`
- Remote root: `/home/jiajia/experiments/zcu104_dma_shell_mbconv_k3_e3_pilot_v1`
- Vivado: `/opt/Xilinx/Vivado/2022.2/bin/vivado`
- Vitis HLS: `/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls`
- Device: `xczu7ev-ffvc1156-2-e`
- Board part: `xilinx.com:zcu104:part0:1.1`
- PL clock: 200 MHz / 5.0 ns
- Expected VLNV: `hwnas.org:hls:zcu104_stage4_k3_e3_kernel:1.0`

Authentication must be typed interactively. The password must not be placed in
these commands, scripts, shell history substitutions, Git, or logs.

## Exact manual server commands

The local bundle is named `zcu104_dma_shell_mbconv_k3_e3_pilot_v1.bundle`.
After it has been uploaded to
`/home/jiajia/experiments/zcu104_dma_shell_mbconv_k3_e3_pilot_v1/staging/`, log
in and run:

```bash
set -euo pipefail

pilot_root=/home/jiajia/experiments/zcu104_dma_shell_mbconv_k3_e3_pilot_v1
bundle="$pilot_root/staging/zcu104_dma_shell_mbconv_k3_e3_pilot_v1.bundle"
repo="$pilot_root/repo"

mkdir -p "$pilot_root/staging"
test -f "$bundle"
git bundle verify "$bundle"

if [ ! -d "$repo/.git" ]; then
  git clone "$bundle" "$repo"
fi

git -C "$repo" fetch "$bundle" \
  codex/zcu104-dma-shell-v1:refs/remotes/pilot/codex/zcu104-dma-shell-v1
git -C "$repo" switch -C codex/zcu104-dma-shell-v1 \
  refs/remotes/pilot/codex/zcu104-dma-shell-v1
test -z "$(git -C "$repo" status --porcelain)"

cd "$repo"
bash hardware/zcu104_dma_shell_v1/remote/run_mbconv_k3_e3_pilot.sh
```

The runner executes, in order: HLS synthesis/export, `component.xml` and VLNV
validation, Block Design creation and `validate_bd_design`, project synthesis,
route implementation, WNS/TNS recording, the non-negative timing gate, then
bitstream and XSA generation. If WNS is negative or TNS is nonzero, it records
the route evidence and stops before bitstream/XSA.

## Expected evidence

All generated evidence remains under:

```text
results/zcu104_dma_shell_mbconv_k3_e3_pilot_v1/
```

Important files are `hls/export_result.json`, `component_identity.json`,
`vivado/timing_gate.json`, `vivado/build_result.json`, the timing/utilization
reports, routed DCP, `.bit`, `.xsa`, logs, and `sha256sums.txt`.

None of these files constitutes board execution, PS/DMA end-to-end latency, or
power evidence.
