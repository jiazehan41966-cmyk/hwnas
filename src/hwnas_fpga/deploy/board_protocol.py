"""Binary protocol for dynamic validation-set inference over UART."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import BinaryIO


REQUEST_MAGIC = b"\xA5\x5A"
RESPONSE_MAGIC = b"\x5A\xA5"
PROTOCOL_VERSION = 1

CMD_LOAD_RUN = 0x01
CMD_RUN_REPEAT = 0x02
CMD_PING = 0x7F

REQUEST_HEADER = struct.Struct("<2sBBII")
RESPONSE_BODY = struct.Struct("<BBBII8bBII")


@dataclass(frozen=True)
class BoardRequest:
    command: int
    sample_id: int
    payload: bytes


@dataclass(frozen=True)
class BoardResponse:
    status: int
    command: int
    sample_id: int
    cycles: int
    logits: tuple[int, ...]
    argmax: int
    checksum: int
    repeat_count: int
    frame_crc32: int | None = field(default=None, compare=False)


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def encode_request(request: BoardRequest) -> bytes:
    if not 0 <= request.command <= 255:
        raise ValueError("command must fit uint8")
    header = REQUEST_HEADER.pack(
        REQUEST_MAGIC,
        PROTOCOL_VERSION,
        int(request.command),
        int(request.sample_id),
        len(request.payload),
    )
    body = header[2:] + request.payload
    return header + request.payload + struct.pack("<I", crc32(body))


def decode_request(frame: bytes) -> BoardRequest:
    if len(frame) < REQUEST_HEADER.size + 4:
        raise ValueError("request frame is truncated")
    magic, version, command, sample_id, payload_len = REQUEST_HEADER.unpack(
        frame[: REQUEST_HEADER.size]
    )
    if magic != REQUEST_MAGIC:
        raise ValueError("bad request magic")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported request version {version}")
    expected_size = REQUEST_HEADER.size + payload_len + 4
    if len(frame) != expected_size:
        raise ValueError(f"request size {len(frame)} != expected {expected_size}")
    body = frame[2:-4]
    observed_crc = struct.unpack("<I", frame[-4:])[0]
    if crc32(body) != observed_crc:
        raise ValueError("request CRC mismatch")
    return BoardRequest(
        command=command,
        sample_id=sample_id,
        payload=frame[REQUEST_HEADER.size:-4],
    )


def encode_response(response: BoardResponse) -> bytes:
    if len(response.logits) != 8:
        raise ValueError("response must contain exactly 8 logits")
    body = RESPONSE_BODY.pack(
        PROTOCOL_VERSION,
        int(response.status),
        int(response.command),
        int(response.sample_id),
        int(response.cycles),
        *(int(value) for value in response.logits),
        int(response.argmax),
        int(response.checksum),
        int(response.repeat_count),
    )
    return RESPONSE_MAGIC + body + struct.pack("<I", crc32(body))


def decode_response(frame: bytes) -> BoardResponse:
    expected_size = 2 + RESPONSE_BODY.size + 4
    if len(frame) != expected_size:
        raise ValueError(f"response size {len(frame)} != expected {expected_size}")
    if frame[:2] != RESPONSE_MAGIC:
        raise ValueError("bad response magic")
    body = frame[2:-4]
    observed_crc = struct.unpack("<I", frame[-4:])[0]
    if crc32(body) != observed_crc:
        raise ValueError("response CRC mismatch")
    unpacked = RESPONSE_BODY.unpack(body)
    version, status, command, sample_id, cycles = unpacked[:5]
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported response version {version}")
    logits = tuple(int(value) for value in unpacked[5:13])
    argmax, checksum, repeat_count = unpacked[13:]
    return BoardResponse(
        status=int(status),
        command=int(command),
        sample_id=int(sample_id),
        cycles=int(cycles),
        logits=logits,
        argmax=int(argmax),
        checksum=int(checksum),
        repeat_count=int(repeat_count),
        frame_crc32=int(observed_crc),
    )


def read_response(stream: BinaryIO, *, max_scan_bytes: int = 4096) -> BoardResponse:
    prefix = bytearray()
    for _ in range(max_scan_bytes):
        value = stream.read(1)
        if not value:
            continue
        prefix += value
        if len(prefix) > 2:
            prefix = prefix[-2:]
        if bytes(prefix) == RESPONSE_MAGIC:
            remaining = RESPONSE_BODY.size + 4
            payload = bytearray()
            while len(payload) < remaining:
                block = stream.read(remaining - len(payload))
                if not block:
                    raise TimeoutError("response frame truncated after magic")
                payload.extend(block)
            return decode_response(RESPONSE_MAGIC + payload)
    raise TimeoutError("response magic not found")


def repeat_payload(repeat_count: int) -> bytes:
    if repeat_count <= 0:
        raise ValueError("repeat_count must be positive")
    return struct.pack("<I", int(repeat_count))
