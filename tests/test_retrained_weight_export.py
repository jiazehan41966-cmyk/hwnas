import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "hls_lut_builder"
    / "board_harness"
    / "scripts"
    / "export_retrained_weights_to_e2e_mem.py"
)

spec = importlib.util.spec_from_file_location("export_retrained_weights_to_e2e_mem", SCRIPT_PATH)
exporter = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = exporter
spec.loader.exec_module(exporter)


class RetrainedWeightExportTests(unittest.TestCase):
    def test_fold_conv_bn_matches_manual_formula(self) -> None:
        state = {
            "conv.weight": torch.tensor([[[[1.0]]], [[[2.0]]]]),
            "conv.bias": torch.tensor([0.5, -0.5]),
            "bn.weight": torch.tensor([2.0, 0.5]),
            "bn.bias": torch.tensor([0.25, -0.25]),
            "bn.running_mean": torch.tensor([0.1, -0.2]),
            "bn.running_var": torch.tensor([4.0, 0.25]),
            "bn.num_batches_tracked": torch.tensor(7),
        }
        consumed: set[str] = set()

        weight, bias = exporter.fold_conv_bn(state, "conv", "bn", consumed)

        inv_std = state["bn.weight"] / torch.sqrt(state["bn.running_var"] + 1e-5)
        expected_weight = state["conv.weight"] * inv_std.reshape(-1, 1, 1, 1)
        expected_bias = (state["conv.bias"] - state["bn.running_mean"]) * inv_std + state["bn.bias"]
        torch.testing.assert_close(weight, expected_weight)
        torch.testing.assert_close(bias, expected_bias)
        self.assertEqual(set(state), consumed)

    def test_quantize_symmetric_is_reproducible(self) -> None:
        tensor = torch.tensor([-1.0, 0.0, 1.0])

        q1, scale1 = exporter.quantize_symmetric(tensor, bit_width=8)
        q2, scale2 = exporter.quantize_symmetric(tensor, bit_width=8)

        self.assertEqual(scale1, scale2)
        self.assertEqual(scale1, 1.0 / 127.0)
        self.assertEqual(q1.tolist(), [-127, 0, 127])
        self.assertEqual(q2.tolist(), [-127, 0, 127])

    def test_partitioned_bank_mem_order_and_padding(self) -> None:
        spec = {
            "param_interfaces": {
                "weights": {
                    "mode": "partitioned_bram",
                    "banks": [
                        {"prefix": "w_00", "indices": [0, 0], "data_width": 8},
                        {"prefix": "w_01", "indices": [0, 1], "data_width": 8},
                        {"prefix": "w_10", "indices": [1, 0], "data_width": 8},
                        {"prefix": "w_11", "indices": [1, 1], "data_width": 8},
                    ],
                    "data_width": 8,
                }
            },
            "case_meta": {"weights_depth": 12},
            "constants": {"OUT_CH": 3, "IN_CH": 4},
        }
        tensor = torch.arange(12, dtype=torch.int32).reshape(3, 4)

        with tempfile.TemporaryDirectory() as tmp:
            role_export = exporter.write_tensor_mem(
                data_dir=Path(tmp),
                role="unit_role",
                kind="weights",
                tensor=tensor,
                spec=spec,
            )
            banks = {
                Path(file_info["mem_file"]).name: Path(file_info["mem_file"]).read_text(encoding="utf-8").splitlines()
                for file_info in role_export.files
            }

        self.assertEqual(banks["unit_role_w_00.mem"], ["00", "02", "08", "0a"])
        self.assertEqual(banks["unit_role_w_01.mem"], ["01", "03", "09", "0b"])
        self.assertEqual(banks["unit_role_w_10.mem"], ["04", "06", "00", "00"])
        self.assertEqual(banks["unit_role_w_11.mem"], ["05", "07", "00", "00"])


if __name__ == "__main__":
    unittest.main()
