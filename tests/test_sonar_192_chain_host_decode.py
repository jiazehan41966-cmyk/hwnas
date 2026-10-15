"""Functional raw6 geometry/transport checks; fixtures are not accuracy evidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.sonar_192_chain import host_decode as h
from tools.sonar_192_chain.contract import ROOT


@pytest.fixture(scope="module")
def model():
    from tools.sonar_yolov9_baseline.author_adapter import _load_author_modules
    _load_author_modules(ROOT / "reference/yolov9")
    return SimpleNamespace(training=False, head_type="quality_aware", _author_adapter_main_only=True,
                           head=SimpleNamespace(nc=2, nl=4, reg_max=1, no=6,
                                                stride=torch.tensor([4., 8., 16., 32.])))


def raw_fixture():
    raw = []
    for size in (160, 80, 40, 20):
        value = torch.zeros(1, 6, size, size)
        value[:, :4] = torch.tensor([1., 2., 3., 4.]).reshape(1, 4, 1, 1)
        value[:, 4] = -2.
        value[:, 5] = 2.
        raw.append(value)
    return raw


def test_anchor_grid_stride_and_class_sigmoid(model):
    raw = raw_fixture()
    decoded, returned = h.native_predictions_from_raw(model, raw)
    assert decoded.shape == (1, 6, 34000)
    offset = 0
    for stride, size in zip((4, 8, 16, 32), (160, 80, 40, 20)):
        # Anchor (0.5, 0.5), ltrb=(1,2,3,4) -> center=(1.5,1.5), wh=(4,6).
        assert decoded[0, :4, offset].tolist() == [1.5*stride, 1.5*stride, 4.*stride, 6.*stride]
        assert decoded[0, 0, offset+1].item() == 2.5*stride
        assert decoded[0, 1, offset+size].item() == 2.5*stride
        offset += size*size
    assert torch.equal(decoded[0, 4:, 0], torch.sigmoid(torch.tensor([-2., 2.])))
    assert all(a is b for a, b in zip(raw, returned))


@pytest.mark.parametrize("fault", ("swapped", "negative_distance", "nonfinite", "wrong_channels"))
def test_reject_wrong_output_contract(model, fault):
    raw = raw_fixture()
    if fault == "swapped":
        raw[0], raw[1] = raw[1], raw[0]
    elif fault == "negative_distance":
        raw[0][0, 0, 0, 0] = -1
    elif fault == "nonfinite":
        raw[0][0, 5, 0, 0] = float("nan")
    else:
        raw[0] = raw[0][:, :5]
    with pytest.raises(ValueError, match="semantics"):
        h.native_predictions_from_raw(model, raw)


@pytest.mark.parametrize("bits", (8, 16))
def test_integer_output_scale_dtype_and_level_order(bits):
    nodes, outputs, values = [], {}, {}
    for index, level in enumerate(h.LEVELS):
        name = "raw." + level
        outputs[level] = name
        nodes.append(dict(id=name, output_bits=bits, output_exp=index-3, output_shape=[6, 1, 1]))
        values[name] = np.full((1, 6, 1, 1), index+1, dtype="int"+str(bits))
    graph = dict(nodes=nodes, outputs=outputs)
    raw = h.dequantize_integer_outputs(graph, values)
    assert [tensor[0, 0, 0, 0].item() for tensor in raw] == [.125, .5, 1.5, 4.]
    values[outputs["P2"]] = values[outputs["P2"]].astype(np.int32)
    with pytest.raises(ValueError, match="dtype"):
        h.dequantize_integer_outputs(graph, values)
