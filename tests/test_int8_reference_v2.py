import unittest

import torch
import torch.nn as nn

from hwnas_fpga.deploy.fixed_point import (
    FixedPointContract,
    quantize_bias_int32,
    requantize_per_output_int8,
)
from hwnas_fpga.deploy.int8_reference import (
    UnsupportedIntegerOperatorError,
    run_integer_reference,
)
from hwnas_fpga.deploy.quantization import QuantizationConfig, build_quantized_weight_package


class IntegerReferenceV2Tests(unittest.TestCase):
    def _model(self):
        return nn.Sequential(
            nn.Conv2d(1, 2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2, 2),
        ).eval()

    def test_contract_declares_bias_accumulator_and_requantization(self):
        contract = FixedPointContract().to_dict()
        self.assertEqual(contract["schema_version"], 2)
        self.assertEqual(contract["bias_dtype"], "int32")
        self.assertEqual(contract["accumulator_dtype"], "int32")
        self.assertEqual(contract["rounding"], "nearest_even")
        self.assertEqual(contract["requantization"], "per_output_tensor_scale")

    def test_bias_and_per_output_requantization_are_integer(self):
        bias = quantize_bias_int32(
            torch.tensor([0.5, -0.5]), input_scale=0.25, weight_scale=0.5
        )
        self.assertEqual(bias.dtype, torch.int32)
        result = requantize_per_output_int8(
            torch.tensor([[100, -100]], dtype=torch.int64),
            input_scale=0.5,
            weight_scale=[0.5, 0.25],
            output_scale=0.5,
            channel_axis=1,
        )
        self.assertEqual(result.dtype, torch.int8)
        self.assertEqual(result.tolist(), [[50, -25]])

    def test_layer_trace_has_auditable_hashes(self):
        model = self._model()
        package, summary = build_quantized_weight_package(
            model, config=QuantizationConfig(quantize_bias=True)
        )
        self.assertEqual(package["schema_version"], 2)
        result = run_integer_reference(model, torch.randn(2, 1, 8, 8), package)
        self.assertEqual(tuple(result.output_int8.shape), (2, 2))
        self.assertTrue(result.layer_traces)
        for row in result.layer_traces:
            self.assertIn("layer", row)
            self.assertIn("input_kind", row)
            self.assertEqual(len(row["simulator_sha256"]), 64)
            self.assertEqual(len(row["quantization_spec_sha256"]), 64)

    def test_unsupported_operator_fails_closed(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid()).eval()
        package, _ = build_quantized_weight_package(
            model, config=QuantizationConfig(quantize_bias=True)
        )
        with self.assertRaises(UnsupportedIntegerOperatorError):
            run_integer_reference(model, torch.randn(1, 2), package)


if __name__ == "__main__":
    unittest.main()
