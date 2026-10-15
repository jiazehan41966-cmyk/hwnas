"""Directed mathematical witnesses and frozen INT8 oracle cross-checks."""
import unittest

import numpy as np

from tools.fpga_h2 import numeric as frozen_int8
from tools.hardware_measurement_pilot import reference as ref


class IntegerReferenceTests(unittest.TestCase):
    def test_positive_negative_half_ties_and_shift_zero(self):
        values = np.array([-7, -5, -3, -1, 0, 1, 3, 5, 7], dtype=np.int64)
        np.testing.assert_array_equal(ref.round_shift_nearest_even(values, 1), [-4, -2, -2, 0, 0, 0, 2, 2, 4])
        np.testing.assert_array_equal(ref.round_shift_nearest_even(values, 0), values)
        extremes = np.array([-(1 << 63), (1 << 63) - 1], dtype=np.int64)
        np.testing.assert_array_equal(ref.round_shift_nearest_even(extremes, 31), [-(1 << 32), 1 << 32])

    def test_signed_saturation_boundaries_both_precisions(self):
        for bits, dtype in [(8, np.int8), (16, np.int16)]:
            low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            values = np.array([low - 1, low, low + 1, high - 1, high, high + 1], dtype=np.int64)
            result = ref.requantize(values, 1, 0, bits=bits)
            np.testing.assert_array_equal(result, [low, low, low + 1, high - 1, high, high])
            self.assertEqual(result.dtype, dtype)

    def test_wrap_is_exact_at_int32_and_int64_extremes(self):
        values = np.array([-(1 << 63), (1 << 63) - 1, -(1 << 32) - 1,
                           -(1 << 31) - 1, -(1 << 31), (1 << 31) - 1, 1 << 31, 1 << 32], dtype=np.int64)
        expected = [((int(value) + (1 << 31)) % (1 << 32)) - (1 << 31) for value in values]
        np.testing.assert_array_equal(ref.wrap_int32(values), expected)

    def test_int16_products_and_bias_wrap(self):
        x = np.full((2, 1, 1), -32768, dtype=np.int16)
        weight = np.full((1, 2, 1, 1), -32768, dtype=np.int16)
        actual = ref.conv_accumulator(x, weight, [0], input_bits=16, weight_bits=16)
        np.testing.assert_array_equal(actual, [[[-2147483648]]])
        actual = ref.conv_accumulator(x, weight, [-2147483648], input_bits=16, weight_bits=16)
        np.testing.assert_array_equal(actual, [[[0]]])
        output = ref.pointwise_conv(x, weight, [0], input_bits=16, weight_bits=16,
                                    output_bits=16, multiplier=1, shift=16)
        np.testing.assert_array_equal(output, [[[-32768]]])
        weight[0, 1, 0, 0] = 32767
        np.testing.assert_array_equal(ref.conv_accumulator(x, weight, [0], input_bits=16, weight_bits=16), [[[32768]]])

    def test_maximum_multiplier_product_uses_int64(self):
        result = ref.requantize(np.array([-(1 << 31), (1 << 31) - 1], dtype=np.int64), (1 << 31) - 1, 31, bits=16)
        np.testing.assert_array_equal(result, [-32768, 32767])

    def test_add_rounds_each_unequal_scale_input_before_sum(self):
        # Staged result: round_even(1/2) + 1 = 1; one fused rounding would give 2.
        a, b = np.array([[[1, -1, 3, -3]]], dtype=np.int8), np.ones((1, 1, 4), dtype=np.int8)
        result = ref.staged_add(a, b, a_scale=(1, 1), b_scale=(1, 0), output_scale=(1, 0))
        np.testing.assert_array_equal(result, [[[1, 1, 3, -1]]])
        # A scale that expands the first input also proves intermediate saturation.
        result = ref.staged_add(np.array([[[100]]], dtype=np.int8), np.array([[[-100]]], dtype=np.int8),
            a_scale=(2, 0), b_scale=(1, 0), output_scale=(1, 0))
        np.testing.assert_array_equal(result, [[[27]]])

    def test_concat_order_and_individual_requantization(self):
        a = np.array([[[1, 3]], [[5, 7]]], dtype=np.int8)
        b = np.array([[[-1, -3]]], dtype=np.int8)
        result = ref.requant_concat([a, b], [(1, 1), (2, 0)])
        np.testing.assert_array_equal(result, [[[0, 2]], [[2, 4]], [[-2, -6]]])

    def test_nearest2x_preserves_edge_and_channel_locations(self):
        values = np.array([[[1, 2], [3, 4]], [[-32768, 32767], [-1, 0]]], dtype=np.int16)
        result = ref.nearest2x(values, bits=16)
        self.assertEqual(result.shape, (2, 4, 4))
        for channel in range(2):
            for row in range(2):
                for column in range(2):
                    np.testing.assert_array_equal(result[channel, 2 * row:2 * row + 2, 2 * column:2 * column + 2],
                                                  np.full((2, 2), values[channel, row, column], dtype=np.int16))

    def test_int8_crosscheck_existing_conv_and_requant(self):
        rng = np.random.default_rng(20261012)
        for depthwise, kernel, stride, padding in [(False, 1, 1, 0), (True, 3, 1, 1), (True, 5, 2, 2)]:
            x = rng.integers(-128, 128, size=(3, 6, 7), dtype=np.int8)
            shape = (3, 1, kernel, kernel) if depthwise else (4, 3, 1, 1)
            weight = rng.integers(-128, 128, size=shape, dtype=np.int8)
            bias = rng.integers(-200000, 200001, size=shape[0], dtype=np.int32)
            expected = frozen_int8.conv_acc(x, weight, bias, stride, padding, 3 if depthwise else 1)
            actual = ref.conv_accumulator(x, weight, bias, stride=stride, padding=padding, depthwise=depthwise)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(ref.requantize(actual, 13, 14), frozen_int8.requantize_int32_to_int8(expected, 13, 14))

    def test_int8_crosscheck_existing_aligned_add(self):
        rng = np.random.default_rng(20261012)
        a = rng.integers(-128, 128, size=(3, 4, 5), dtype=np.int8)
        b = rng.integers(-128, 128, size=a.shape, dtype=np.int8)
        for ea, eb, eo in [(-3, -1, 0), (2, -2, 1), (0, 0, 0)]:
            common = max(ea, eb)
            actual = ref.staged_add(a, b, a_scale=frozen_int8.ratio(ea, common),
                b_scale=frozen_int8.ratio(eb, common), output_scale=frozen_int8.ratio(common, eo))
            np.testing.assert_array_equal(actual, frozen_int8.aligned_add(a, b, ea, eb, eo))

    def test_depthwise_matches_independent_python_integer_loops(self):
        rng = np.random.default_rng(919)
        for kernel in (3, 5):
            x = rng.integers(-32768, 32768, (2, 3, 4), dtype=np.int16)
            weight = rng.integers(-32768, 32768, (2, 1, kernel, kernel), dtype=np.int16)
            bias = np.array([2147483647, -2147483648], dtype=np.int32)
            pad = kernel // 2
            expected = np.empty(x.shape, dtype=np.int64)
            for c in range(2):
                for y in range(3):
                    for z in range(4):
                        total = int(bias[c])
                        for dy in range(kernel):
                            for dz in range(kernel):
                                yy, zz = y + dy - pad, z + dz - pad
                                if 0 <= yy < 3 and 0 <= zz < 4:
                                    total += int(x[c, yy, zz]) * int(weight[c, 0, dy, dz])
                        expected[c, y, z] = (total + (1 << 31)) % (1 << 32) - (1 << 31)
            actual = ref.conv_accumulator(x, weight, bias, input_bits=16, weight_bits=16, padding=pad, depthwise=True)
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_numeric_inputs_fail_closed(self):
        for value in [np.array([1.5]), np.array([True]), np.array([1], dtype=np.uint64)]:
            with self.assertRaises(ValueError):
                ref.requantize(value, 1, 0)
        for multiplier in [0, -1, 1 << 31, 1.0, True]:
            with self.assertRaises(ValueError):
                ref.requantize([1], multiplier, 0)
        for shift in [-1, 32, 1.5, False]:
            with self.assertRaises(ValueError):
                ref.requantize([1], 1, shift)
        for values in [[1 << 31], [-(1 << 31) - 1]]:
            with self.assertRaises(ValueError):
                ref.requantize(values, 1, 0)
        with self.assertRaises(ValueError):
            ref.requantize([1], 1, 0, bits=32)

    def test_invalid_geometry_and_range_fail_closed(self):
        x = np.ones((2, 3, 4), dtype=np.int8)
        weight = np.ones((3, 2, 1, 1), dtype=np.int8)
        calls = [
            lambda: ref.conv_accumulator(x, weight, [0, 0]),
            lambda: ref.conv_accumulator(x, weight[:, :1], [0, 0, 0]),
            lambda: ref.conv_accumulator(x, weight, [0, 0, 0], stride=0),
            lambda: ref.conv_accumulator(x, weight, [0, 0, 0], padding=-1),
            lambda: ref.conv_accumulator(x, np.ones((2, 2, 3, 3), dtype=np.int8), [0, 0], depthwise=True),
            lambda: ref.conv_accumulator(x, np.ones((2, 1, 5, 5), dtype=np.int8), [0, 0], depthwise=True),
            lambda: ref.nearest2x(np.ones((1, 2), dtype=np.int8)),
            lambda: ref.nearest2x(np.ones((1, 0, 2), dtype=np.int8)),
            lambda: ref.nearest2x(np.array([[[128]]], dtype=np.int16), bits=8),
            lambda: ref.staged_add(x, x[:, :1], a_scale=(1, 0), b_scale=(1, 0), output_scale=(1, 0)),
            lambda: ref.requant_concat([x, x[:, :1]], [(1, 0), (1, 0)]),
            lambda: ref.requant_concat([x, x], [(1, 0)]),
        ]
        for operation in calls:
            with self.assertRaises(ValueError):
                operation()


if __name__ == "__main__":
    unittest.main()
