# Stage4 MBConv-K3-E3 pilot core

This directory preserves the generated operator-cost source for
`stage4__mbconv_k3_e3__32x28x28_to_32x28x28__s1` and changes only its control
interface: the original HLS AXI4-Lite return port becomes direct
`ap_ctrl_hs`, because the frozen DMA wrapper owns the software-visible control
register bank. The external weights and biases BRAM interfaces are constrained
to `ram_1p`; this leaves the other physical port of each true-dual-port BRAM
available for PS preload through its AXI BRAM Controller.

The calculation body, INT8/INT32 types, pixel-major stream order, residual,
weight layout, bias layout, saturation, and ReLU6 behavior are unchanged.
Weights remain a 7,008-byte BRAM master and biases remain a 224-element INT32
BRAM master (896 bytes). The single-port storage constraint may change parameter
preload scheduling, but it does not change any arithmetic or tensor layout.

Run only on the authorized 2022.2 ZCU104 server:

```bash
/opt/Xilinx/Vitis_HLS/2022.2/bin/vitis_hls \
  -f hardware/zcu104_dma_shell_v1/hls/mbconv_k3_e3/run_hls.tcl
```
