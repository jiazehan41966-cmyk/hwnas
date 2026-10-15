"""Explicit raw-output DAG for the registered 192v2 detector.

Every learned convolution is taken from the selected model. Functional blur,
residuals, PAN, SiLU, distance ReLU and unequal-width concat stay explicit.
This graph ends at four six-channel raw heads; host decode/NMS is separate.
"""
from __future__ import annotations

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from hwnas_fpga.models.builder import SkipBlock

from .contract import digest, model_contract

LEVELS = ("P2", "P3", "P4", "P5")
GRAPH_VERSION = "sonar192v2_raw6_graph_v1"


def build_graph(model):
    if getattr(model, "search_space_version", None) != "sonar_backbone_neck192_v2":
        raise ValueError("graph requires the registered sonar_backbone_neck192_v2 model")
    contract = model_contract(model.candidate_architecture)
    if model.head.no != 6 or model.head.reg_max != 1 or model.head.nc != 2:
        raise ValueError("six-channel direct-distance head required")
    if not isinstance(model.head.distance_activation, nn.ReLU):
        raise ValueError("direct distances must retain ReLU")
    nodes, edges, conv_names = [], {"image": [1, 640, 640]}, set()

    def primitive(name, op, inputs, shape, **extra):
        if name in edges or any(source not in edges for source in inputs):
            raise ValueError("duplicate node or non-topological source: " + name)
        row = dict(id=name, op=op, inputs=list(inputs),
                   input_shapes=[list(edges[source]) for source in inputs],
                   output_shape=list(shape), **extra)
        nodes.append(row)
        edges[name] = list(shape)
        return name

    def conv(name, source, activation="linear", *, bn=None):
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Conv2d):
            raise ValueError("expected actual Conv2d: " + name)
        if (layer.kernel_size[0] != layer.kernel_size[1] or layer.stride[0] != layer.stride[1]
                or layer.padding[0] != layer.padding[1] or layer.dilation != (1, 1)
                or layer.padding_mode != "zeros" or layer.in_channels != edges[source][0]):
            raise ValueError("unsupported or inconsistent convolution geometry: " + name)
        if bn is not None and not isinstance(model.get_submodule(bn), (nn.BatchNorm2d, nn.Identity)):
            raise ValueError("expected paired BN: " + bn)
        k, s, p = layer.kernel_size[0], layer.stride[0], layer.padding[0]
        c, h, w = edges[source]
        conv_names.add(name)
        return primitive(name, "conv", [source], [layer.out_channels, (h+2*p-k)//s+1, (w+2*p-k)//s+1],
                         module_path=name, Cin=c, Cout=layer.out_channels, H=h, W=w,
                         kernel=k, stride=s, padding=p, dilation=1, groups=layer.groups,
                         activation=activation, bias=layer.bias is not None, bn_folded=bn is not None,
                         bn_path=bn, operator_type="DW" if layer.groups == c == layer.out_channels
                         else "GroupConv" if layer.groups != 1 else "Conv")

    def add(name, left, right, *, residual=False):
        if edges[left] != edges[right]:
            raise ValueError("addition shape mismatch: " + name)
        return primitive(name, "residual_add" if residual else "add", [left, right], edges[left], activation="linear")

    def block(prefix, source):
        layer = model.get_submodule(prefix)
        if isinstance(layer, nn.Identity) or (isinstance(layer, SkipBlock) and not layer.use_conv):
            return source
        if not all(hasattr(layer, field) for field in ("dw_conv", "project_conv", "use_residual")):
            raise ValueError("unsupported MBConv block: " + prefix)
        if (not isinstance(layer.dw_relu, nn.ReLU6)
                or (layer.use_expand and not isinstance(layer.expand_relu, nn.ReLU6))):
            raise ValueError("MBConv nonlinearities changed")
        original = source
        if layer.use_expand:
            source = conv(prefix + ".expand_conv", source, "relu6", bn=prefix + ".expand_bn")
        source = conv(prefix + ".dw_conv", source, "relu6", bn=prefix + ".dw_bn")
        if hasattr(layer, "blur_kernel"):
            kernel = layer.blur_kernel.detach().cpu()
            fixed = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], dtype=kernel.dtype)/16
            if not torch.equal(kernel, fixed.reshape(1, 1, 3, 3).repeat(kernel.shape[0], 1, 1, 1)):
                raise ValueError("fixed blur kernel drift")
            c, h, w = edges[source]
            source = primitive(prefix + ".blur", "blur", [source], [c, (h+1)//2, (w+1)//2],
                               Cin=c, Cout=c, kernel=3, stride=2, padding=1, dilation=1, groups=c,
                               fixed_kernel=[[1, 2, 1], [2, 4, 2], [1, 2, 1]], denominator=16,
                               activation="linear", buffer_path=prefix + ".blur_kernel")
        source = conv(prefix + ".project_conv", source, bn=prefix + ".project_bn")
        return add(prefix + ".residual_add", source, original, residual=True) if layer.use_residual else source

    source = conv("backbone.stem.conv.conv", "image", "relu", bn="backbone.stem.conv.bn")
    source = conv("backbone.stages.0.0.conv", source, "relu", bn="backbone.stages.0.0.bn")
    features = {}
    for stage in range(1, 8):
        for index, layer in enumerate(model.backbone.stages[stage]):
            if not isinstance(layer, nn.Identity):
                source = block(f"backbone.stages.{stage}.{index}", source)
        if stage in model.feature_indices:
            features[LEVELS[model.feature_indices.index(stage)]] = source
    if set(features) != set(LEVELS):
        raise ValueError("four actual backbone features required")
    fused = [conv(f"neck.lateral_convs.{level}.0", features[level], "relu6", bn=f"neck.lateral_convs.{level}.1")
             for level in LEVELS]
    for index in range(2, -1, -1):
        level = LEVELS[index]
        source = primitive(f"neck.top_down.{level}.nearest", "nearest", [fused[index+1]], edges[fused[index]], factor=2)
        source = add(f"neck.top_down.{level}.add", fused[index], source)
        fused[index] = block(f"neck.top_down_blocks.{level}", source)
    outputs = fused
    if model.neck.bottom_up_pan:
        outputs = [fused[0]]
        for index in range(1, 4):
            level = LEVELS[index]
            c, h, w = edges[outputs[-1]]
            source = primitive(f"neck.bottom_up.{level}.maxpool", "maxpool", [outputs[-1]], [c, h//2, w//2],
                               kernel=2, stride=2, padding=0)
            source = conv(f"neck.bottom_up_projections.{level}.0", source, "relu6", bn=f"neck.bottom_up_projections.{level}.1")
            source = add(f"neck.bottom_up.{level}.add", fused[index], source)
            outputs.append(block(f"neck.bottom_up_blocks.{level}", source))
    predictions = {}
    for index, (level, source) in enumerate(zip(LEVELS, outputs)):
        branches = []
        for branch in ("cv2", "cv3"):
            value = source
            for hidden in (0, 1):
                prefix = f"head.{branch}.{index}.{hidden}"
                layer = model.get_submodule(prefix)
                if not isinstance(layer.act, nn.SiLU):
                    raise ValueError("quality hidden activation must remain SiLU")
                expected_channels = (64 if branch == "cv2" else 48)
                expected_input = 48 if hidden == 0 else expected_channels
                expected_groups = 4 if branch == "cv2" and hidden == 1 else 1
                if (layer.conv.in_channels != expected_input or layer.conv.out_channels != expected_channels
                        or layer.conv.kernel_size != (3, 3) or layer.conv.groups != expected_groups
                        or layer.conv.stride != (1, 1)):
                    raise ValueError("registered quality hidden tower geometry changed")
                value = conv(prefix + ".conv", value, bn=prefix + ".bn")
                value = primitive(prefix + ".act", "silu", [value], edges[value], activation="silu")
            value = conv(f"head.{branch}.{index}.2", value)
            predictor = model.get_submodule(f"head.{branch}.{index}.2")
            if (predictor.kernel_size != (1, 1) or predictor.stride != (1, 1)
                    or predictor.groups != (4 if branch == "cv2" else 1)):
                raise ValueError("registered final grouped/pointwise prediction changed")
            if branch == "cv2":
                value = primitive(f"head.distance_relu.{level}", "relu", [value], edges[value], activation="relu")
            branches.append(value)
        if [edges[value][0] for value in branches] != [4, 2]:
            raise ValueError("head concat must preserve 4 distance then 2 quality channels")
        predictions[level] = primitive(f"head.output.{level}", "concat", branches, [6, *edges[branches[0]][1:]],
                                       dim=1, channel_order=["ltrb4", "class_quality2"])
    actual_convs = {name for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)}
    if conv_names != actual_convs:
        raise ValueError("graph omits/adds model convolutions: " + str(sorted(conv_names ^ actual_convs)))
    graph = dict(schema_version=GRAPH_VERSION, space_version=contract["space_version"],
                 architecture=deepcopy(contract["architecture"]), architecture_sha256=contract["architecture_sha256"],
                 architecture_id=contract["architecture_id"], structure_id=contract["structure_id"], num_classes=2,
                 full_contract_sha256=contract["full_contract_sha256"], head_contract_sha256=contract["head_contract_sha256"],
                 input_shape=[1, 640, 640], nodes=nodes, edges=edges, outputs=predictions,
                 output_contract="raw_ltrb_relu4_then_class_quality_logits2_per_P2_P3_P4_P5",
                 decode_scope="CPU_host_anchor_decode_sigmoid_NMS_not_part_of_integer_graph",
                 hardware_verified=False)
    graph["structural_graph_sha256"] = digest(graph)
    return graph


def raw_predictions(model, images):
    """Use native feature/hidden-tower execution without mutable decode caches."""
    _, features = model.extract_features(images)
    return tuple(torch.cat((model.head.distance_activation(model.head.cv2[i](feature)),
                            model.head.cv3[i](feature)), dim=1) for i, feature in enumerate(features))


def float_graph(model, graph, images):
    if images.ndim != 4 or list(images.shape[1:]) != graph["input_shape"]:
        raise ValueError("graph input geometry mismatch")
    values = {"image": images}
    for node in graph["nodes"]:
        sources = [values[name] for name in node["inputs"]]
        op = node["op"]
        if op == "conv":
            value = model.get_submodule(node["module_path"])(sources[0])
            if node["activation"] == "relu":
                value = value.clamp_min(0)
            elif node["activation"] == "relu6":
                value = value.clamp(0, 6)
            elif node["activation"] != "linear":
                raise ValueError("unsupported fused activation")
        elif op in ("add", "residual_add"):
            value = sources[0] + sources[1]
        elif op == "blur":
            value = F.conv2d(sources[0], model.get_buffer(node["buffer_path"]), stride=2, padding=1, groups=node["groups"])
        elif op == "nearest":
            value = F.interpolate(sources[0], size=node["output_shape"][-2:], mode="nearest")
        elif op == "maxpool":
            value = F.max_pool2d(sources[0], 2, 2)
        elif op == "silu":
            value = F.silu(sources[0])
        elif op == "relu":
            value = sources[0].clamp_min(0)
        elif op == "concat":
            value = torch.cat(sources, dim=1)
        else:
            raise ValueError("unsupported primitive: " + op)
        if list(value.shape[1:]) != node["output_shape"]:
            raise ValueError("explicit graph shape mismatch: " + node["id"])
        values[node["id"]] = value
    return values
