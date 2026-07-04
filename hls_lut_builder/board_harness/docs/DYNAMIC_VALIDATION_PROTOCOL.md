# Dynamic validation protocol v1

This protocol is reserved for full-validation-set INT8 inference. It does not
replace the historical 15-byte TX-only latency packet.

## Request

Little-endian layout:

`magic=A5 5A | version:u8 | command:u8 | sample_id:u32 | payload_len:u32 |
payload | crc32:u32`

CRC32 covers `version` through the final payload byte.

Commands:

- `0x01 LOAD_RUN`: payload is one flattened signed-INT8 input tensor.
- `0x02 RUN_REPEAT`: payload is `repeat_count:u32`; reuse the buffered input.
- `0x7f PING`: empty payload.

## Response

`magic=5A A5 | version:u8 | status:u8 | command:u8 | sample_id:u32 |
cycles:u32 | logits:8*int8 | argmax:u8 | checksum:u32 | repeat_count:u32 |
crc32:u32`

CRC32 covers `version` through `repeat_count`.

Board accuracy is claimable only when every expected sample has `status=0`,
matching sample ID, valid CRC, and logits exactly equal to the frozen
bit-exact software reference.
