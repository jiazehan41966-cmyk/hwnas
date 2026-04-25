# Board Harness

This directory contains the first reusable single-operator board-measurement
flow for exported HLS kernels.

## Design Review

The original "UART in / UART out full tensor transport" proposal is directionally
correct, but it is not the best first implementation for operator-level board
measurement. The main corrections in this v1 harness are:

1. Do not stream full tensors over `115200` UART in the first version.
   For operators such as `stem_conv_k3_s2`, one run would otherwise need to
   transfer hundreds of kilobytes, which would dominate wall-clock time and make
   repeated measurement impractical.

2. Use the exported HLS RTL top module directly.
   The exported IP package is useful for catalog packaging, but the simplest
   reusable board harness is a direct RTL wrapper around the generated top
   module, with an internal AXI-Lite master to start and poll the kernel.

3. Treat `component.xml` as an interface description, not a full tensor-layout
   description.
   Bus metadata such as `MEM_WIDTH` is useful, but `MEM_SIZE` is not sufficient
   to recover the real flattened parameter depth for all operators. The harness
   generator therefore combines interface parsing with case/source parsing.

## v1 Measurement Flow

1. Start from an exported HLS case under `hls_lut_builder/results/p/...`.
2. Parse the exported RTL top and `component.xml`.
3. Generate a single-operator harness project:
   - fixed clock/reset shell
   - AXI-Lite control master
   - preinitialized input stream source
   - output checksum sink
   - deterministic weight/bias memories
   - UART result transmitter
4. Build a Vivado project and bitstream for one operator.
5. Program the board and read back `status + cycles + checksum + output word count`.

## What v1 Measures

- Cycles from AXI-Lite `ap_start` write completion to `ap_done` observation
- Output word count observed on the AXIS sink
- A deterministic checksum over output data

This is sufficient for single-operator board timing validation and sanity
checking. It does not yet provide high-throughput dynamic tensor upload.

## UART Result Packet

The v1 harness is TX-only after configuration. Each programmed bitstream emits
exactly one fixed-length UART result packet:

- `magic[0] = 0xA5`
- `magic[1] = 0x5A`
- `status_code`
- `cycles` as little-endian `uint32`
- `checksum` as little-endian `uint32`
- `word_count` as little-endian `uint32`

The exact packet layout is documented in `docs/UART_PROTOCOL.md`.

## Current Scope

The initial scaffold is designed to support the current packed-stream operator
family, starting from:

- `stem_conv_k3_s2`
- `mbconv_e3_k3`

The generator is structured so additional operators can reuse the same harness
shape after their parameter formulas are added.

## Files

- `templates/`: top-level RTL, XDC, and Vivado Tcl templates
- `modules/`: reusable RTL helper modules
- `scripts/generate_harness.py`: generates a concrete board harness project from
  an exported HLS case
- `configs/`: board and harness defaults

## Board Config

The default real-board config is:

- `configs/board_av7k325.yaml`

It uses the AV7K325 differential 200 MHz clock pair, one user key as reset,
the USB-UART pins, and two user LEDs.
