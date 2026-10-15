"""Directed T6 numeric witnesses; these tests do not certify hardware."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from tools.hardware_measurement_pilot import reference as legacy
from tools.hardware_measurement_pilot.inventory import build_inventory
from tools.hardware_template_coverage_v1 import reference as ref


def scalar_conv(x, weights, bias, stride, padding, depthwise=False):
    """Independent Python-int witness, including wrap after all products."""
    cin, h, w = x.shape
    cout, _, k, _ = weights.shape
    oh, ow = (h + 2 * padding - k) // stride + 1, (w + 2 * padding - k) // stride + 1
    result = np.empty((cout, oh, ow), dtype=np.int64)
    for oc in range(cout):
        for y in range(oh):
            for z in range(ow):
                total = int(bias[oc])
                for ic in ([oc] if depthwise else range(cin)):
                    for dy in range(k):
                        for dz in range(k):
                            iy, ix = y * stride + dy - padding, z * stride + dz - padding
                            if 0 <= iy < h and 0 <= ix < w:
                                total += int(x[ic, iy, ix]) * int(weights[oc, 0 if depthwise else ic, dy, dz])
                result[oc, y, z] = (total + (1 << 31)) % (1 << 32) - (1 << 31)
    return result


def scalar_pool(x, kernel, stride, padding):
    cin, h, w = x.shape
    oh, ow = (h + 2 * padding - kernel) // stride + 1, (w + 2 * padding - kernel) // stride + 1
    minimum = np.iinfo(x.dtype).min
    result = np.empty((cin, oh, ow), dtype=x.dtype)
    for c in range(cin):
        for y in range(oh):
            for z in range(ow):
                values = [int(x[c, iy, ix])
                    for dy in range(kernel) for dz in range(kernel)
                    if 0 <= (iy := y * stride + dy - padding) < h
                    and 0 <= (ix := z * stride + dz - padding) < w]
                result[c, y, z] = max(values, default=minimum)
    return result


class ExtendedNumericTests(unittest.TestCase):
    def test_existing_numeric_functions_are_the_same_objects(self):
        for name in ("wrap_int32", "round_shift_nearest_even", "requantize",
                     "pointwise_conv", "depthwise_conv", "nearest2x", "staged_add"):
            self.assertIs(getattr(ref, name), getattr(legacy, name))

    def test_standard3_matches_python_ints_on_stride_borders_and_wrap(self):
        rng = np.random.default_rng(20261013)
        for bits in (8, 16):
            low, high = -(1 << (bits - 1)), 1 << (bits - 1)
            x = rng.integers(low, high, (2, 5, 7), dtype=np.int32).astype(np.int8 if bits == 8 else np.int16)
            weights = rng.integers(-128, 128, (3, 2, 3, 3), dtype=np.int8)
            bias = np.array([2**31 - 1, -2**31, 3], dtype=np.int32)
            for stride, pad in ((1, 1), (2, 1), (2, 0)):
                actual = ref.conv_accumulator(x, weights, bias, input_bits=bits, stride=stride, padding=pad)
                np.testing.assert_array_equal(actual, scalar_conv(x, weights, bias, stride, pad))
        x = np.full((2, 3, 3), -32768, dtype=np.int16)
        weights = np.full((2, 2, 3, 3), -128, dtype=np.int8)
        bias = np.array([2**31 - 1, -2**31], dtype=np.int32)
        actual = ref.conv_accumulator(x, weights, bias, input_bits=16, padding=1)
        np.testing.assert_array_equal(actual, scalar_conv(x, weights, bias, 1, 1))
        self.assertLess(int(actual[0, 1, 1]), 0)

    def test_standard3_requantization_ties_and_saturation(self):
        x = np.zeros((1, 1, 9), dtype=np.int8)
        weights = np.zeros((9, 1, 3, 3), dtype=np.int8)
        bias = np.array([-7, -5, -3, -1, 1, 3, 5, 7, 1000], dtype=np.int32)
        actual = ref.standard_conv3x3(x, weights, bias, multiplier=1, shift=1)
        np.testing.assert_array_equal(actual[:, 0, 0], [-4, -2, -2, 0, 0, 2, 2, 4, 127])

    def test_existing_pointwise_and_depthwise_outputs_remain_identical(self):
        rng = np.random.default_rng(19)
        for depthwise, k, stride, pad in ((False, 1, 1, 0), (True, 3, 2, 1), (True, 5, 1, 2)):
            x = rng.integers(-32768, 32768, (3, 5, 7), dtype=np.int16)
            weights = rng.integers(-128, 128, (3, 1 if depthwise else 3, k, k), dtype=np.int8)
            bias = np.array([2**31 - 1, -2**31, -1], dtype=np.int32)
            args = dict(input_bits=16, stride=stride, padding=pad, depthwise=depthwise)
            np.testing.assert_array_equal(ref.conv_accumulator(x, weights, bias, **args),
                                          legacy.conv_accumulator(x, weights, bias, **args))

    def test_maxpool_negative_padding_and_int_minimum(self):
        for dtype, bits in ((np.int8, 8), (np.int16, 16)):
            x = np.array([[[-9, -8, -7, -6, -5], [-4, -3, -2, -1, -11],
                           [-12, -13, -14, -15, -16]]], dtype=dtype)
            x[0, 0, 0] = np.iinfo(dtype).min
            for k, stride, pad in ((2, 2, 0), (3, 2, 1), (3, 1, 2)):
                actual = ref.maxpool(x, k, stride, pad, bits=bits)
                np.testing.assert_array_equal(actual, scalar_pool(x, k, stride, pad))
                self.assertTrue(np.all(actual < 0))
            all_min = np.full((2, 1, 1), np.iinfo(dtype).min, dtype=dtype)
            np.testing.assert_array_equal(ref.maxpool(all_min, 3, 2, 1, bits=bits), all_min)

    def test_new_operations_reject_invalid_geometry_and_types(self):
        x = np.ones((2, 3, 5), dtype=np.int8)
        weights = np.ones((2, 2, 3, 3), dtype=np.int8)
        calls = [
            lambda: ref.conv_accumulator(x, weights[:, :1], [0, 0]),
            lambda: ref.conv_accumulator(x, weights, [0]),
            lambda: ref.conv_accumulator(x, weights, [0, 0], depthwise=1),
            lambda: ref.conv_accumulator(x, weights, [0, 0], stride=True),
            lambda: ref.conv_accumulator(x, weights, [0, 0], padding=-1),
            lambda: ref.conv_accumulator(x, np.ones((2, 2, 2, 2), dtype=np.int8), [0, 0]),
            lambda: ref.standard_conv3x3(x, np.ones((2, 2, 1, 1), dtype=np.int8), [0, 0], multiplier=1, shift=0),
            lambda: ref.maxpool(x, 0), lambda: ref.maxpool(x, 2, 0),
            lambda: ref.maxpool(x, 10, 1, 0), lambda: ref.maxpool(x, 2, 2, -1),
            lambda: ref.maxpool(x.astype(np.float64)),
            lambda: ref.maxpool(np.array([[[128]]], dtype=np.int16)),
        ]
        for call in calls:
            with self.assertRaises(ValueError):
                call()


class FixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = build_inventory()
        cls.rows = cls.inventory["primitive_inventory"]

    def test_all_65_rows_all_modes_with_small_spatial_witnesses(self):
        """Keep real channels/precision/activation, reduce geometry only here."""
        self.assertEqual(len(self.rows), 65)
        for original in self.rows:
            entry = deepcopy(original)
            shape = entry["shape"]
            op = ref.operator_name(entry)
            h, w = (5, 7) if op != "maxpool" else (6, 8)
            for value in shape["input_shapes"]:
                value[1:] = [h, w]
            if op in ("pw1x1", "conv3x3", "dw3", "dw5", "maxpool"):
                k, s, p = (shape[key] for key in ("kernel", "stride", "padding"))
                output = [(h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1]
                shape.update(H=h, W=w)
            elif op == "nearest2x":
                output = [h * 2, w * 2]
            else:
                output = [h, w]
            shape["output_shape"][1:] = output
            seed = 20261013 + int(entry["primitive_id"][:8], 16)
            for mode in ref.fixture_modes:
                with self.subTest(primitive=entry["primitive_id"][:12], mode=mode):
                    fixture = ref.make_fixture(entry, mode, seed)
                    self.assertEqual(set(fixture), {"a", "b", "w", "bias", "params", "expected"})
                    self.assertEqual(fixture["a"].shape, tuple(shape["input_shapes"][0]))
                    self.assertEqual(fixture["expected"].shape, tuple(shape["output_shape"]))
                    self.assertEqual(fixture["expected"].dtype.itemsize * 8, entry["precision"]["output_bitwidth"])
                    self.assertEqual(int(fixture["params"][6]), ref.ACTIVATION_CODES[shape.get("activation", "linear")])
                    if shape.get("activation") == "relu6":
                        self.assertTrue(np.all((fixture["expected"] >= 0) & (fixture["expected"] <= 12)))
                    elif shape.get("activation") == "relu":
                        self.assertTrue(np.all(fixture["expected"] >= 0))

    def test_real_stem_and_three_pool_shapes_are_never_downscaled(self):
        rows = [row for row in self.rows if row["family"] in ("conv3x3", "maxpool")]
        self.assertEqual(len(rows), 4)
        for entry in rows:
            f = ref.make_fixture(entry, "overflow_extremes", 20261013)
            self.assertEqual(f["a"].shape, tuple(entry["shape"]["input_shapes"][0]))
            self.assertEqual(f["expected"].shape, tuple(entry["shape"]["output_shape"]))
            if entry["family"] == "maxpool":
                self.assertTrue(np.all(f["expected"] == -128))

    def test_seed_mode_pair_and_prepared_case_alias_are_identical(self):
        entry = deepcopy(next(row for row in self.rows if row["family"] == "maxpool"))
        precision = entry.pop("precision")
        prepared = deepcopy(entry)
        prepared.update(operator="maxpool", input_bits=precision["input_bitwidth"],
                        output_bits=precision["output_bitwidth"], weight_bits=None)
        original = dict(entry, precision=precision)
        for mode in ref.fixture_modes:
            one = ref.make_fixture(original, mode, 17)
            two = ref.make_fixture(prepared, mode, 17)
            for key in one:
                self.assertEqual(one[key].tobytes(), two[key].tobytes())
        self.assertNotEqual(ref.make_fixture(original, "random_linear", 17)["a"].tobytes(),
                            ref.make_fixture(original, "random_linear", 18)["a"].tobytes())

    def test_fixture_generator_rejects_inconsistent_shape_precision_activation(self):
        entry = next(row for row in self.rows if row["family"] == "maxpool")
        for path, value in (("output_shape", [48, 1, 1]), ("activation", "relu"),
                            ("stride", 0), ("kernel", True)):
            bad = deepcopy(entry)
            bad["shape"][path] = value
            with self.assertRaises(ValueError):
                ref.make_fixture(bad, "random_linear", 0)
        bad = deepcopy(entry)
        bad["precision"]["weight_bitwidth"] = 8
        with self.assertRaises(ValueError):
            ref.make_fixture(bad, "random_linear", 0)
        with self.assertRaises(ValueError):
            ref.make_fixture(entry, "missing", 0)
        for seed in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                ref.make_fixture(entry, "random_linear", seed)


if __name__ == "__main__":
    unittest.main()
