"""Known lane ordering, tails and reversible integration with HLS case files."""
from pathlib import Path

import numpy as np
import pytest

from tools.detector_v0_2.layout import (
    pack_activation, unpack_activation, pack_weights, unpack_weights,
)


def test_known_signed_lanes_and_spatial_order():
    tensor = np.array([[[-128, 127]], [[-1, 1]]], dtype=np.int8)
    packed = pack_activation(tensor)
    assert packed == bytes([128, 255] + [0]*14 + [127, 1] + [0]*14)
    np.testing.assert_array_equal(unpack_activation(packed, (2, 1, 2)), tensor)


@pytest.mark.parametrize("channels", [1, 21, 32, 48, 192])
def test_activation_roundtrip_including_prediction_channel_tail(channels):
    tensor = np.random.default_rng(42).integers(-128, 128, (channels, 7, 5), dtype=np.int16).astype(np.int8)
    packed = pack_activation(tensor)
    assert len(packed) == 7*5*((channels+15)//16)*16
    np.testing.assert_array_equal(unpack_activation(packed, tensor.shape), tensor)


@pytest.mark.parametrize("shape,groups", [((21,48,1,1),1), ((32,1,3,3),1), ((48,1,3,3),48), ((32,24,1,1),1)])
def test_kernel_roundtrip_and_known_tap_order(shape, groups):
    weights = np.arange(np.prod(shape), dtype=np.int32).reshape(shape).astype(np.int8)
    packed = pack_weights(weights, groups=groups)
    np.testing.assert_array_equal(unpack_weights(packed, shape, groups=groups), weights)
    if groups > 1:
        assert packed[:16] == weights[0,0].tobytes() + bytes(7)
    else:
        expected = weights[0,:,0,0].tobytes() + bytes((-shape[1])%16)
        assert packed[:len(expected)] == expected


def test_nonzero_padding_and_truncated_payloads_are_rejected():
    image = np.zeros((21,2,3), dtype=np.int8)
    packed = bytearray(pack_activation(image)); packed[31] = 1
    with pytest.raises(ValueError, match="padding"):
        unpack_activation(bytes(packed), image.shape)
    with pytest.raises(ValueError, match="byte count"):
        unpack_activation(pack_activation(image)[:-1], image.shape)
    weight = np.zeros((21,48,1,1), dtype=np.int8)
    with pytest.raises(ValueError, match="byte count"):
        unpack_weights(pack_weights(weight)[:-1], weight.shape)
    dw = np.zeros((48,1,3,3), dtype=np.int8)
    packed = bytearray(pack_weights(dw, groups=48)); packed[15] = 1
    with pytest.raises(ValueError, match="padding"):
        unpack_weights(bytes(packed), dw.shape, groups=48)


def test_invalid_types_and_unsupported_grouped_convolution_are_rejected():
    with pytest.raises(ValueError, match="INT8"):
        pack_activation(np.zeros((1,2,3), dtype=np.uint8))
    with pytest.raises(ValueError, match="groups"):
        pack_weights(np.zeros((4,2,3,3), dtype=np.int8), groups=2)
    with pytest.raises(ValueError, match="integer"):
        unpack_activation(bytes(16), (True,1,1))


def test_generated_hls_witness_inputs_and_weights_are_reversible():
    # This checks the actual frozen case files produced by the independent HLS
    # preparer, rather than using this module to manufacture its own expectation.
    import json
    root = Path(__file__).resolve().parents[1]
    case_root = root / "results/detector_v0_2_execution_20261002/hardware/operator_v1"
    if not case_root.is_dir():
        pytest.skip("prepared standalone HLS witnesses are absent")
    names = ("pw48", "prediction21", "dw48", "dw48_stride2", "stem32")
    for index, name in enumerate(names):
        folder = case_root / name
        case = json.loads((folder / "case.json").read_text())
        cfg = case["config"]
        rng = np.random.default_rng(4200 + index)
        assert len(case["tests"]) == 5
        for item in case["tests"]:
            ci, co, k = cfg["cin"], cfg["cout"], cfg["kernel"]
            wi = 1 if cfg["groups"] == ci == co else ci
            image = rng.integers(-128, 128, (ci,item["height"],item["width"]), dtype=np.int16).astype(np.int8)
            weight = rng.integers(-128, 128, (co,wi,k,k), dtype=np.int16).astype(np.int8)
            rng.integers(-20000, 20001, co, dtype=np.int32)  # consume the preparer's bias draw
            if item["stress"] == "overflow_positive":
                image.fill(127); weight.fill(127)
            elif item["stress"] == "overflow_negative":
                image.fill(-128); weight.fill(127)
            original_image = (folder / (item["id"] + "_input.bin")).read_bytes()
            original_weight = (folder / (item["id"] + "_weights.bin")).read_bytes()
            assert pack_activation(image) == original_image
            assert pack_weights(weight, groups=cfg["groups"]) == original_weight
            np.testing.assert_array_equal(unpack_activation(original_image, image.shape), image)
            np.testing.assert_array_equal(unpack_weights(original_weight, weight.shape, groups=cfg["groups"]), weight)
