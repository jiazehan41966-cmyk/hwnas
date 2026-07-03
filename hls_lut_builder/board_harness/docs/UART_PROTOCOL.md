# UART Result Protocol

The v1 single-operator board harness emits one fixed-length UART packet after a
single auto-start run completes.

## Frame layout

Total length: `15` bytes

| Byte(s) | Field | Type | Description |
| --- | --- | --- | --- |
| `0` | `magic0` | `uint8` | Fixed `0xA5` |
| `1` | `magic1` | `uint8` | Fixed `0x5A` |
| `2` | `status_code` | `uint8` | Run result code |
| `3..6` | `cycles` | `uint32_le` | Measured cycles from AXI-Lite start to `ap_done` |
| `7..10` | `checksum` | `uint32_le` | Rolling checksum over observed AXIS output words |
| `11..14` | `word_count` | `uint32_le` | Number of AXIS output words accepted by the sink |

## Status codes

| Code | Meaning |
| --- | --- |
| `0` | OK: kernel control completed and sink observed the expected output count |
| `1` | Sink incomplete: control completed but the sink did not finish normally |
| `2` | Control error: AXI-Lite start/poll sequence reported an error |

## Measurement behavior

- The harness is currently **TX-only**.
- After configuration and reset release, the harness waits for the boot delay,
  auto-starts the kernel once, and transmits one result packet.
- Repeating a measurement currently requires reprogramming the FPGA or adding a
  future re-trigger path.

## Practical collection rule

The host script should:

1. Open the serial port.
2. Clear stale bytes.
3. Program the FPGA bitstream.
4. Wait for the `0xA5 0x5A` header.
5. Read the remaining `13` bytes and decode the frame.
