# PS control handoff (frozen interface, runtime not yet selected)

The v1 shell freezes a polling-oriented control ABI but does not choose a Linux,
PYNQ, or bare-metal implementation. The authoritative offsets are in
`zcu104_dma_shell_v1.h` and `../contracts/platform_contract.json`.

For one Stage4 Skip transaction, allocate physically DMA-accessible, at least
64-byte-aligned input and output buffers of exactly 25,088 bytes. Linearize the
logical `[C,H,W] = [32,28,28]` tensor as HWC bytes: byte offset
`((h * 28 + w) * 32 + c)` stores signed INT8 channel `c`. Flush the input cache,
invalidate the output cache, arm S2MM, clear/start the shell, then arm MM2S.
Completion requires both DMA channels to complete and shell STATUS.DONE to be
set with ERROR_FLAGS equal to zero.

Read each 64-bit counter with a high-low-high retry so rollover cannot tear the
value. `OPERATOR_CORE_LATENCY_CYCLES` is multiplied by `1 / CLOCK_HZ`; it is not
the PS/DMA end-to-end latency. The latter must be recorded as a separate
software timestamp interval once a real PS runtime is authorized and executed.
