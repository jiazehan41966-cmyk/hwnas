"""Frozen host decoding for the four direct-distance raw6 output tensors.

Integer deployment stops at raw tensors.  This adapter applies the exact author
anchor geometry and the project's already-frozen sigmoid/top-k/class-wise NMS.
It does not convert a software witness into hardware or accuracy evidence.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
from torch import nn

from .contract import ROOT, sha256_file, verify_frozen_sources

LEVELS = ("P2", "P3", "P4", "P5")
POSTPROCESSING = dict(score_threshold=0.0, nms_iou_threshold=0.5,
                      pre_nms_topk=500, max_detections=100, image_size=640)


def _head(model):
    if model.training or getattr(model, "head_type", None) != "quality_aware":
        raise ValueError("Host decode requires the frozen quality-aware model in eval mode")
    head = model.head
    if (head.nc, head.nl, head.reg_max, head.no) != (2, 4, 1, 6):
        raise ValueError("Host decode supports only four-scale direct raw6 heads")
    if head.stride.detach().cpu().tolist() != [4.0, 8.0, 16.0, 32.0]:
        raise ValueError("Host decode stride contract changed")
    if not getattr(model, "_author_adapter_main_only", False):
        raise ValueError("Host decode requires the main-only direct head")
    return head


def native_predictions_from_raw(model, raw):
    """Return native ``(Bx6x34000 decoded, four raw heads)`` without loss/NMS."""
    head = _head(model)
    if not isinstance(raw, (tuple, list)) or len(raw) != 4:
        raise ValueError("Expected raw P2/P3/P4/P5 in that order")
    batch, device, dtype = raw[0].shape[0], raw[0].device, raw[0].dtype
    for stride, tensor in zip((4, 8, 16, 32), raw):
        if (tensor.ndim != 4 or tuple(tensor.shape) != (batch, 6, 640 // stride, 640 // stride)
                or tensor.device != device or tensor.dtype != dtype
                or not tensor.is_floating_point() or not torch.isfinite(tensor).all()
                or (tensor[:, :4] < 0).any()):
            raise ValueError("Raw head geometry/dtype/direct-distance semantics differ")
    geometry = importlib.import_module("utils.tal.anchor_generator")
    anchors, strides = (value.transpose(0, 1) for value in
                        geometry.make_anchors(raw, head.stride.to(device=device, dtype=dtype), 0.5))
    box, classes = torch.cat([tensor.reshape(batch, 6, -1) for tensor in raw], dim=2).split((4, 2), dim=1)
    decoded_box = geometry.dist2bbox(box, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    return torch.cat((decoded_box, classes.sigmoid()), dim=1), list(raw)


def decode_raw(model, raw):
    """Apply the frozen author/project host path to dequantized raw outputs."""
    from tools.sonar_yolov9_baseline.author_adapter import decode_author
    return decode_author(model, native_predictions_from_raw(model, raw), **POSTPROCESSING)


def dequantize_integer_outputs(graph, outputs):
    """Interpret signed raw head bytes with the exported per-tensor exponents."""
    import numpy as np
    nodes = {node["id"]: node for node in graph["nodes"]}
    raw = []
    for level in LEVELS:
        name = graph["outputs"][level]
        node, value = nodes[name], np.asarray(outputs[name])
        if (value.dtype != np.dtype("int" + str(node["output_bits"])) or value.ndim != 4
                or list(value.shape[1:]) != node["output_shape"]):
            raise ValueError("Raw integer output dtype or shape differs from graph")
        raw.append(torch.from_numpy(value.copy()).float() * (2.0 ** node["output_exp"]))
    return tuple(raw)


def source_contract():
    """Pin the actual geometry and NMS implementation used by this adapter."""
    verify_frozen_sources()
    geometry = importlib.import_module("utils.tal.anchor_generator")
    from tools.sonar_yolov9_baseline import author_adapter
    from hwnas_fpga.training import detection_trainer
    paths = [Path(__file__), Path(geometry.__file__), Path(author_adapter.__file__),
             Path(detection_trainer.__file__), ROOT / "tools/sonar_head_c01b/model.py"]
    return dict(schema_version="sonar192_host_decode_v1", precision="torch_float32_host",
                raw_semantics="P2_P3_P4_P5_ltrb_relu4_class_quality_logits2",
                anchor_offset=0.5, strides=[4, 8, 16, 32], postprocessing=POSTPROCESSING,
                source_sha256={str(path.resolve()): sha256_file(path) for path in paths},
                hardware_execution=False)


class Raw6Adapter(nn.Module):
    """Bridge a quantized raw source to the existing evaluator output contract."""

    def __init__(self, native_model, raw_source):
        super().__init__()
        _head(native_model)
        self.native_model, self.raw_source = native_model, raw_source

    def forward(self, images):
        if self.training:
            raise ValueError("Raw6Adapter is inference-only; call eval()")
        raw = self.raw_source(images)["raw_predictions"]
        return {"predictions": native_predictions_from_raw(self.native_model, raw)}
