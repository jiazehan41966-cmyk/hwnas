import io
import unittest

from hwnas_fpga.deploy.board_protocol import (
    BoardRequest,
    BoardResponse,
    CMD_LOAD_RUN,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    read_response,
    repeat_payload,
)


class RequestTests(unittest.TestCase):
    def test_request_roundtrip(self) -> None:
        request = BoardRequest(CMD_LOAD_RUN, 42, bytes(range(32)))
        self.assertEqual(decode_request(encode_request(request)), request)

    def test_crc_failure_is_rejected(self) -> None:
        frame = bytearray(encode_request(BoardRequest(CMD_LOAD_RUN, 1, b"abc")))
        frame[-1] ^= 0x01
        with self.assertRaises(ValueError):
            decode_request(bytes(frame))

    def test_repeat_count_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            repeat_payload(0)


class ResponseTests(unittest.TestCase):
    def test_response_roundtrip(self) -> None:
        response = BoardResponse(
            status=0,
            command=CMD_LOAD_RUN,
            sample_id=9,
            cycles=123,
            logits=(-4, -3, -2, -1, 0, 1, 2, 3),
            argmax=7,
            checksum=99,
            repeat_count=1,
        )
        frame = encode_response(response)
        self.assertEqual(decode_response(frame), response)
        self.assertEqual(read_response(io.BytesIO(b"noise" + frame)), response)


if __name__ == "__main__":
    unittest.main()
