"""Frozen V0.1 prediction-only ONNX interface; no hardware readiness claim."""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace


class PredictionOutputs(nn.Module):
    def __init__(self, candidate):
        super().__init__()
        self.candidate = candidate

    def forward(self, image):
        return tuple(self.candidate(image)['predictions'])


def fold_candidate_bn(candidate):
    """Copy an eval candidate and reuse the canonical Conv/BN folding path."""
    if candidate.training or any(m.training for m in candidate.modules() if isinstance(m, nn.BatchNorm2d)):
        raise ValueError('BN folding requires eval mode and frozen running statistics')
    folded = deepcopy(candidate)
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(m, nn.BatchNorm2d) for m in folded.modules()):
        raise ValueError('Unfolded BN remains: unsupported adjacency')
    return folded, pairs


def export_candidate_predictions(candidate, output_path, example):
    """Export static batch-1 NCHW raw logits in increasing-stride level order.

    Decoding/NMS remains outside this interface. No precision conversion occurs.
    """
    if tuple(example.shape) != (1, 1, 640, 640):
        raise ValueError('Export contract is static [1,1,640,640]')
    if candidate.training:
        raise ValueError('Export requires eval mode')
    levels = tuple(candidate.levels)
    if levels not in [('P3', 'P4', 'P5'), ('P2', 'P3', 'P4'), ('P2', 'P3', 'P4', 'P5')]:
        raise ValueError('Unsupported V0.1 output combination')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    wrapper = PredictionOutputs(candidate).eval()
    torch.onnx.export(wrapper, example, str(path), input_names=['image'],
                      output_names=list(levels), opset_version=13,
                      dynamo=False, do_constant_folding=True, dynamic_axes=None)
    import onnx
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    outputs = [{'name': p, 'stride': int(s), 'shape': [1, 21, 640 // s, 640 // s]}
               for p, s in zip(levels, candidate.feature_strides)]
    observed = [{'name': o.name, 'shape': [d.dim_value for d in o.type.tensor_type.shape.dim]}
                for o in graph.graph.output]
    if observed != [{'name': o['name'], 'shape': o['shape']} for o in outputs]:
        raise ValueError(f'ONNX outputs differ from contract: {observed}')
    metadata = {'schema_version': 1, 'outputs': outputs, 'layout': 'NCHW',
                'dtype': 'float32', 'raw_logits': True, 'decode_nms': 'external',
                'channel_order': 'anchor_major: [tx,ty,tw,th,obj,class0,class1]',
                'anchors': candidate.anchor_sizes,
                'architecture': candidate.candidate_architecture,
                'hardware_parity': 'not_verified'}
    entry = graph.metadata_props.add()
    entry.key, entry.value = 'sonar_v0_1_contract', json.dumps(metadata)
    onnx.save(graph, str(path))
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
