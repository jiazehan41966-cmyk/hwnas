"""Primitive cost topology of the NEW four-scale ELASTIC neck space.

The graph is an independent structural contract, not a PTQ export or a hardware
measurement. BN parameters count toward software Params; deployment folds BN
into its preceding convolution. Adds, residuals, nearest interpolation, and
pooling remain explicit so an incomplete cost library cannot silently omit them.
No old 144-candidate detector graph is imported or generated here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
BACKBONE_SPEC = ROOT / "configs/search_space_sonar_v0_1.yaml"
BACKBONE_SPEC_SHA256 = "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4"
LEVELS = ("P2", "P3", "P4", "P5")
FEATURE_CHANNELS = (24, 32, 96, 320)
SPATIAL = (160, 80, 40, 20)
DEFAULT_PRECISION = dict(input_bitwidth=8, weight_bitwidth=8,
                         accumulator_bitwidth=32, output_bitwidth=8)
GRAPH_VERSION = "modnas_elastic_primitive_640_v1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def validate_architecture(architecture: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(architecture, Mapping) or set(architecture) != {"pan", "td_kernels", "pan_kernels"}:
        raise ValueError("NEW architecture requires pan, td_kernels, pan_kernels only")
    pan = architecture["pan"]
    if type(pan) is not bool:
        raise ValueError("pan must be bool")
    td, bottom = architecture["td_kernels"], architecture["pan_kernels"]
    if not isinstance(td, (list, tuple)) or len(td) != 3:
        raise ValueError("td_kernels must contain P2/P3/P4 kernels")
    if not isinstance(bottom, (list, tuple)) or len(bottom) != (3 if pan else 0):
        raise ValueError("inactive PAN must have no kernels; active PAN requires P3/P4/P5")
    if any(type(k) is not int or k not in (3, 5) for k in (*td, *bottom)):
        raise ValueError("kernel must be integer 3 or 5")
    return dict(pan=pan, td_kernels=list(td), pan_kernels=list(bottom))


def primitive_shape(node: Mapping[str, Any]) -> dict[str, Any]:
    """Complete NEW primitive identity, including independent input/output sizes."""
    keys = ("op", "operator_type", "input_shapes", "output_shape", "Cin", "Cout", "H", "W",
            "kernel", "stride", "padding", "groups", "activation", "bias", "bn_folded",
            "interpolation", "alignment", "expanded_channels")
    return {key: deepcopy(node[key]) for key in keys if key in node}


def h3_shape(node: Mapping[str, Any]) -> dict[str, Any] | None:
    """Old evidence can certify exact convolution primitives, never NEW topology."""
    if node["op"] != "conv":
        return None
    return {key: node[key] for key in ("operator_type", "Cin", "Cout", "H", "W", "kernel",
                                     "stride", "groups", "activation", "expanded_channels")}


def build_cost_graph(architecture: Mapping[str, Any], *, num_classes: int = 2) -> dict[str, Any]:
    arch = validate_architecture(architecture)
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if hashlib.sha256(BACKBONE_SPEC.read_bytes()).hexdigest() != BACKBONE_SPEC_SHA256:
        raise ValueError("Fixed B1 backbone specification SHA256 changed")
    nodes, edges = [], {"image": [1, 640, 640]}

    def conv(name, source, cout, kernel=1, stride=1, groups=1, activation="relu6", bn=True, bias=False):
        cin, height, width = edges[source]
        padding = kernel // 2
        output = [cout, (height + 2 * padding - kernel) // stride + 1,
                  (width + 2 * padding - kernel) // stride + 1]
        node = dict(id=name, op="conv", operator_type="DW" if groups != 1 else "Conv",
                    inputs=[source], input_shapes=[list(edges[source])], output_shape=output,
                    Cin=cin, Cout=cout, H=height, W=width, kernel=kernel, stride=stride,
                    padding=padding, groups=groups, activation=activation, bias=bias,
                    bn_folded=bn, expanded_channels=0,
                    software_parameters=cout * (cin // groups) * kernel * kernel
                    + (cout if bias else 0) + (2 * cout if bn else 0))
        nodes.append(node)
        edges[name] = output
        return name

    def primitive(name, op, sources, output, **extra):
        node = dict(id=name, op=op, operator_type=op, inputs=list(sources),
                    input_shapes=[list(edges[source]) for source in sources], output_shape=list(output),
                    software_parameters=0, **extra)
        nodes.append(node)
        edges[name] = list(output)
        return name

    def add(name, left, right, residual=False):
        if edges[left] != edges[right]:
            raise ValueError("Addition requires equal shapes")
        return primitive(name, "residual_add" if residual else "add", (left, right), edges[left],
                         activation="linear", alignment="requant_each_input_to_common_scale_pending")

    def mbconv(prefix, source, cout, kernel, stride, residual=False):
        hidden = edges[source][0] * 3
        expanded = conv(prefix + ".expand_conv", source, hidden)
        depthwise = conv(prefix + ".dw_conv", expanded, hidden, kernel, stride, hidden)
        projected = conv(prefix + ".project_conv", depthwise, cout, activation="linear")
        return add(prefix + ".residual_add", projected, source, True) if residual else projected

    # Fixed complete B1 backbone, rebuilt from its frozen stage contract.
    source = conv("backbone.stem.conv.conv", "image", 32, 3, 2, activation="relu")
    source = conv("backbone.stages.0.0.conv", source, 16, activation="relu")
    features = {}
    for stage, cout, kernel, stride in ((1, 24, 3, 2), (2, 32, 5, 2), (4, 64, 5, 2),
                                        (5, 96, 5, 1), (6, 160, 5, 2), (7, 320, 5, 1)):
        source = mbconv(f"backbone.stages.{stage}.0", source, cout, kernel, stride)
        if stage in (1, 2, 5, 7):
            # Stage3 is an identity; P3 retains the stage2 tensor.
            features[{1: "P2", 2: "P3", 5: "P4", 7: "P5"}[stage]] = source
    fused = [conv(f"neck.lateral_convs.{level}.0", features[level], 48) for level in LEVELS]
    for index in range(2, -1, -1):
        level = LEVELS[index]
        interpolated = primitive(f"neck.top_down.{level}.nearest", "nearest", (fused[index + 1],),
                                 edges[fused[index]], interpolation="nearest_exact_factor_two")
        merged = add(f"neck.top_down.{level}.add", fused[index], interpolated)
        fused[index] = mbconv(f"neck.top_down_blocks.{level}", merged, 48, arch["td_kernels"][index], 1, True)
    outputs = [fused[0]]
    if arch["pan"]:
        for index in range(1, 4):
            level = LEVELS[index]
            shape = edges[outputs[-1]]
            pooled = primitive(f"neck.bottom_up.{level}.maxpool", "maxpool", (outputs[-1],),
                               [48, shape[1] // 2, shape[2] // 2], kernel=2, stride=2, padding=0,
                               activation="linear")
            projected = conv(f"neck.bottom_up_projections.{level}.0", pooled, 48)
            merged = add(f"neck.bottom_up.{level}.add", fused[index], projected)
            outputs.append(mbconv(f"neck.bottom_up_blocks.{level}", merged, 48,
                                  arch["pan_kernels"][index - 1], 1, True))
    else:
        outputs = fused
    predictions = {level: conv(f"head.towers.{index}.0", value, 3 * (5 + num_classes),
                               activation="linear", bn=False, bias=True)
                   for index, (level, value) in enumerate(zip(LEVELS, outputs))}
    for node in nodes:
        node["primitive_shape_sha256"] = canonical_hash(primitive_shape(node))
    graph = dict(schema_version=1, version=GRAPH_VERSION, architecture=arch,
                 architecture_sha256=canonical_hash(arch), num_classes=num_classes,
                 input_shape=[1, 640, 640], nodes=nodes, edges=edges, outputs=predictions,
                 Params=sum(node["software_parameters"] for node in nodes),
                 backbone_spec_sha256=BACKBONE_SPEC_SHA256,
                 graph_scope="structural_cost_topology_not_integer_export_not_hardware_measurement",
                 quantization_status="NEW_NECK_RESIDUAL_PAN_INTEGER_CONTRACT_NOT_VERIFIED",
                 hardware_search_status="BLOCKED_MISSING_NEW_SPACE_COST")
    graph["cost_graph_sha256"] = canonical_hash(graph)
    return graph


__all__ = ["ROOT", "DEFAULT_PRECISION", "canonical_hash", "validate_architecture", "primitive_shape",
           "h3_shape", "build_cost_graph"]
