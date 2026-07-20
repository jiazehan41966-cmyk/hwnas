import unittest

import torch
import torch.nn.functional as F

from hwnas_fpga.deploy.fixed_point import (
    FixedPointContract,
    conv2d_int_reference,
    linear_int_reference,
    requantize_int8,
    round_divide_nearest_even,
    saturate_signed,
)


class RoundingTests(unittest.TestCase):
    def test_ties_round_to_even(self) -> None:
        values = torch.tensor([1, 3, 5, -1, -3, -5])
        result = round_divide_nearest_even(values, 2)
        self.assertEqual(result.tolist(), [0, 2, 2, 0, -2, -2])

    def test_requantize_saturates(self) -> None:
        result = requantize_int8(
            torch.tensor([1000, -1000]),
            multiplier_numerator=1,
            multiplier_denominator=1,
        )
        self.assertEqual(result.tolist(), [127, -128])


class IntegerReferenceTests(unittest.TestCase):
    def test_linear_matches_integer_matmul(self) -> None:
        inputs = torch.tensor([[1, 2, 3]], dtype=torch.int8)
        weights = torch.tensor([[2, 0, -1]], dtype=torch.int8)
        output = linear_int_reference(inputs, weights, torch.tensor([4]))
        self.assertEqual(output.tolist(), [[3]])

    def test_conv_matches_float_for_small_exact_integers(self) -> None:
        inputs = torch.tensor(
            [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
            dtype=torch.int8,
        )
        weights = torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.int8)
        expected = F.conv2d(inputs.float(), weights.float()).to(torch.int64)
        actual = conv2d_int_reference(inputs, weights)
        self.assertTrue(torch.equal(actual, expected))

    def test_accumulator_saturation(self) -> None:
        values = torch.tensor([-20, 0, 20])
        self.assertEqual(saturate_signed(values, 4).tolist(), [-8, 0, 7])


class ContractTests(unittest.TestCase):
    def test_contract_is_explicit(self) -> None:
        contract = FixedPointContract().to_dict()
        self.assertEqual(contract["rounding"], "nearest_even")
        self.assertEqual(contract["accumulator_width"], 32)


if __name__ == "__main__":
    unittest.main()
