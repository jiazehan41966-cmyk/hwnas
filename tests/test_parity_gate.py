import unittest

from hwnas_fpga.deploy.parity_gate import audit_parity_records


def _record(kind: str, mismatch: int = 0):
    return {
        "input_kind": kind,
        "sample_id": 1,
        "layer": "stage1.block0",
        "element_count": 128,
        "mismatch_count": mismatch,
        "simulator_sha256": "a" * 64,
        "hls_testbench_sha256": "b" * 64,
    }


class ParityGateTests(unittest.TestCase):
    def test_all_input_kinds_and_zero_mismatch_pass(self) -> None:
        result = audit_parity_records(
            [
                _record("real_sample"),
                _record("boundary_tensor"),
                _record("random_tensor"),
            ],
            quantization_contract="per_tensor_symmetric_int8_v1",
        )
        self.assertTrue(result["overall_pass"])

    def test_one_mismatch_blocks_board_validation(self) -> None:
        result = audit_parity_records(
            [
                _record("real_sample", mismatch=1),
                _record("boundary_tensor"),
                _record("random_tensor"),
            ],
            quantization_contract="per_tensor_symmetric_int8_v1",
        )
        self.assertFalse(result["overall_pass"])
        self.assertFalse(result["gates"]["zero_integer_mismatch"])

    def test_missing_boundary_class_fails(self) -> None:
        result = audit_parity_records(
            [_record("real_sample"), _record("random_tensor")],
            quantization_contract="per_tensor_symmetric_int8_v1",
        )
        self.assertFalse(result["gates"]["boundary_tensors_present"])


if __name__ == "__main__":
    unittest.main()
