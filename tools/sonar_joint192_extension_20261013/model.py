"""Isolated controlled extensions of joint192; no old space or operator edits.

The configuration is a base joint192 architecture plus explicit changes.  This
module does not enumerate a search space, run training, or admit FPGA costs.
DMDC/CFEM are channel adaptations of the local paper-derived implementations,
not claims of a newly verified reproduction of the original papers.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.models.builder import (
    MBConvBlock, DMDCScaleBranch, CFEMScaleAttention, CFEMCBAMWeight,
)
from hwnas_fpga.models.sonar_op import ReparamDualRFDepthwise
from hwnas_fpga.models.sonar_operator_pool_p0 import FrequencyLiteDepthwise
from tools.detector_elastic_pilot.model import ElasticFourScaleNeck, _hash_named_tensors
from tools.sonar_joint192_validation.model import (
    StaticJoint192Detector, BlurDownMBConv, DenseDecimateMBConv,
    REFERENCE_ARCHITECTURES as BASE_REFERENCES,
    validate_architecture as validate_base_architecture,
    architecture_hash as base_architecture_hash,
    architecture_id as base_architecture_id,
)


SEARCH_SPACE_VERSION = "sonar_joint192_extension_20261013_v1"
INITIALIZATION_POLICY = "JOINT192_EXTENSION_SEMANTIC_MATCHED_INITIALIZATION_V1"
DEFAULT_CHANGES = dict(stage1_spatial="keep", stage2_spatial="keep",
                       p2_op="keep", p2_inputs="adjacent", p2_fusion="sum")
REFERENCE_ARCHITECTURES = {
    name: dict(base_architecture=deepcopy(base), changes=deepcopy(DEFAULT_CHANGES))
    for name, base in BASE_REFERENCES.items()
}
_CHOICES = {
    "stage1_spatial": ("keep", "sonarop", "frequencylite"),
    "stage2_spatial": ("keep", "sonarop", "frequencylite"),
    "p2_op": ("keep", "identity", "mbconv_k3", "mbconv_k5", "sonarop",
              "frequencylite", "dmdc48", "cfem48"),
    "p2_inputs": ("adjacent", "lateral_only", "direct_p4"),
    "p2_fusion": ("sum", "concat_project", "gated_sum"),
}


def validate_architecture(architecture: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(architecture, Mapping) or set(architecture) != {
            "base_architecture", "changes"}:
        raise ValueError("extension requires exactly base_architecture and changes")
    base = validate_base_architecture(architecture["base_architecture"])
    changes = architecture["changes"]
    if not isinstance(changes, Mapping) or set(changes) != set(_CHOICES):
        raise ValueError("changes requires all five explicit extension fields")
    for name, choices in _CHOICES.items():
        if type(changes[name]) is not str or changes[name] not in choices:
            raise ValueError(f"{name} requires one of {choices}")
    if changes["p2_inputs"] == "lateral_only" and changes["p2_fusion"] != "sum":
        raise ValueError("single-input P2 must pass through with sum; no dead fusion parameters")
    if (changes["p2_op"] in ("sonarop", "frequencylite")
            and base["neck"]["td_ops"]["P2"] != "mbconv_k3"):
        raise ValueError("P2 spatial probes require the original mbconv_k3 reference")
    return dict(base_architecture=base,
                changes={name: changes[name] for name in _CHOICES})


def architecture_hash(architecture: Mapping[str, Any]) -> str:
    value = json.dumps(validate_architecture(architecture), sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def architecture_id(architecture: Mapping[str, Any]) -> str:
    return f"joint192_extension_{architecture_hash(architecture)[:16]}"


def _semantic_seed(seed: int, location: str) -> int:
    encoded = f"{INITIALIZATION_POLICY}:{seed}:{location}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)


def _new_module(seed: int, location: str, factory):
    """Allocate on CPU without consuming caller or data-loader RNG state."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(_semantic_seed(seed, location))
        return factory()


def _copy_matching_state(target: nn.Module, reference: nn.Module) -> None:
    previous = reference.state_dict()
    target.load_state_dict({name: previous[name] if name in previous
                            and previous[name].shape == value.shape else value
                            for name, value in target.state_dict().items()}, strict=True)


def _replace_spatial(block: MBConvBlock, operation: str, seed: int, location: str) -> None:
    if operation == "keep":
        return
    reference = block.dw_conv
    if not isinstance(reference, nn.Conv2d) or reference.groups != reference.in_channels:
        raise ValueError("spatial probe requires an original depthwise convolution")
    channels, stride = reference.in_channels, reference.stride[0]
    if reference.stride != (stride, stride):
        raise ValueError("spatial probe requires square stride")
    if operation == "sonarop":
        replacement = _new_module(seed, f"{location}.sonarop",
                                  lambda: ReparamDualRFDepthwise(channels, stride))
        # Pair the existing receptive-field branch only.  The other branch is
        # independently initialized, not fabricated by reshaping old weights.
        branch = replacement.branch3 if reference.kernel_size == (3, 3) else replacement.branch5
        if branch[0].weight.shape != reference.weight.shape:
            raise ValueError("SonarOp pairing supports only an original DW3 or DW5")
        branch[0].load_state_dict(reference.state_dict(), strict=True)
        branch[1].load_state_dict(block.dw_bn.state_dict(), strict=True)
        block.dw_bn, block.dw_relu = nn.Identity(), nn.Identity()
    elif operation == "frequencylite":
        replacement = _new_module(seed, f"{location}.frequencylite",
                                  lambda: FrequencyLiteDepthwise(channels, stride))
        if replacement.high_dw.weight.shape == reference.weight.shape:
            replacement.high_dw.load_state_dict(reference.state_dict(), strict=True)
        # This operator is linear: retain the existing external BN and ReLU6.
    else:
        raise ValueError(f"unsupported spatial probe {operation}")
    # The unchanged BlurDownMBConv.downsample queries this compatibility field.
    # Its blur remains after the spatial operator's activation, with stride 2.
    replacement.out_channels = channels
    block.dw_conv = replacement


class DMDCChannelAdapted(nn.Module):
    """Only channel count generalized; the existing DMDC core is unchanged."""
    dilation_rates = (1, 3, 5)

    def __init__(self, channels: int = 48):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.scale_branches = nn.ModuleList(
            DMDCScaleBranch(channels, rate) for rate in self.dilation_rates)
        self.channel_compression = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU6(inplace=True))

    def forward(self, value):
        return self.channel_compression(torch.cat(
            [branch(value) for branch in self.scale_branches], dim=1))


class CFEMChannelAdapted(nn.Module):
    """Only channels generalized; preserve the local CFEM sigmoid semantics."""
    dilation_rates = (2, 3, 5)

    def __init__(self, channels: int = 48):
        super().__init__()
        if type(channels) is not int or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self.channels = channels
        self.context_branches = nn.ModuleList(nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, channels, 3, stride=1, padding=rate,
                      dilation=rate, bias=False)) for rate in self.dilation_rates)
        self.scale_attention = CFEMScaleAttention(channels)
        self.context_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.input_projection = nn.Conv2d(channels, channels, 1, bias=False)
        self.fusion_attention = CFEMCBAMWeight(channels)
        self.output_channel_attention = CFEMCBAMWeight(channels)

    def forward(self, value):
        first, second, third = [branch(value) for branch in self.context_branches]
        alpha, beta, gamma = self.scale_attention(torch.cat((first, second, third), dim=1))
        context = self.context_projection(alpha * first + beta * second + gamma * third)
        original = self.input_projection(value)
        weight = self.fusion_attention(original + context)
        fused = original * weight + context * (1.0 - weight)
        channel = self.output_channel_attention.channel_weight(fused)
        spatial = self.output_channel_attention.spatial_weight(fused * channel)
        return torch.sigmoid(fused * spatial)


class P2SourceFusion(nn.Module):
    """Explicit source fusion; all trainable routes start at an equal sum."""
    def __init__(self, channels: int, sources: int, mode: str):
        super().__init__()
        if sources not in (1, 2, 3) or mode not in ("sum", "concat_project", "gated_sum"):
            raise ValueError("invalid P2 fusion")
        if sources == 1 and mode != "sum":
            raise ValueError("one source must pass through without fusion parameters")
        self.channels, self.sources, self.mode = channels, sources, mode
        if mode == "concat_project":
            self.projection = nn.Conv2d(channels * sources, channels, 1, bias=False)
            with torch.no_grad():
                self.projection.weight.zero_()
                index = torch.arange(channels)
                for source in range(sources):
                    self.projection.weight[index, source * channels + index, 0, 0] = 1.
        elif mode == "gated_sum":
            self.gates = nn.ModuleList(nn.Conv2d(channels, 1, 1, bias=True)
                                       for _ in range(sources))
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    def source_weights(self, values):
        if self.mode != "gated_sum":
            raise ValueError("only gated_sum has input-dependent source weights")
        logits = torch.stack([gate(value) for gate, value in zip(self.gates, values)], dim=1)
        return torch.softmax(logits, dim=1) * self.sources

    def forward(self, values):
        if len(values) != self.sources:
            raise ValueError("P2 source count differs from the frozen fusion contract")
        if any(value.shape != values[0].shape for value in values):
            raise ValueError("P2 fusion sources must have identical shapes")
        if self.sources == 1:
            return values[0]
        if self.mode == "concat_project":
            return self.projection(torch.cat(values, dim=1))
        if self.mode == "gated_sum":
            weights = self.source_weights(values)
            values = [value * weights[:, index] for index, value in enumerate(values)]
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result


class ExtensionNeck(ElasticFourScaleNeck):
    """Reuse existing modules; separately retain L2 and completed T3/T4."""
    def __init__(self, reference, changes, seed):
        # Do not allocate and then discard a second randomly initialized neck.
        nn.Module.__init__(self)
        self.in_channels, self.out_channels = reference.in_channels, reference.out_channels
        self.levels, self.bottom_up_pan = reference.levels, reference.bottom_up_pan
        self.lateral_convs = reference.lateral_convs
        self.top_down_blocks = reference.top_down_blocks
        self.bottom_up_projections = reference.bottom_up_projections
        self.bottom_up_blocks = reference.bottom_up_blocks
        if self.bottom_up_pan:
            self.bottom_up_pool = reference.bottom_up_pool
        self.p2_inputs = changes["p2_inputs"]
        count = {"lateral_only": 1, "adjacent": 2, "direct_p4": 3}[self.p2_inputs]
        self.p2_fusion = _new_module(seed, f"neck.P2.fusion.{changes['p2_fusion']}.{count}",
            lambda: P2SourceFusion(self.out_channels, count, changes["p2_fusion"]))

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("extension neck requires four backbone features")
        lateral = [self.lateral_convs[level](feature)
                   for level, feature in zip(self.levels, features)]
        fused = list(lateral)
        for index in (2, 1):
            merged = lateral[index] + F.interpolate(
                fused[index + 1], size=lateral[index].shape[-2:], mode="nearest")
            fused[index] = self.top_down_blocks[self.levels[index]](merged)
        sources = [lateral[0]]
        if self.p2_inputs in ("adjacent", "direct_p4"):
            sources.append(F.interpolate(fused[1], size=lateral[0].shape[-2:], mode="nearest"))
        if self.p2_inputs == "direct_p4":
            sources.append(F.interpolate(fused[2], size=lateral[0].shape[-2:], mode="nearest"))
        fused[0] = self.top_down_blocks["P2"](self.p2_fusion(sources))
        if not self.bottom_up_pan:
            return tuple(fused)
        outputs = [fused[0]]
        for index in range(1, 4):
            level = self.levels[index]
            reduced = self.bottom_up_projections[level](self.bottom_up_pool(outputs[-1]))
            if reduced.shape[-2:] != fused[index].shape[-2:]:
                raise ValueError("PAN requires exact factor-two feature shapes")
            outputs.append(self.bottom_up_blocks[level](fused[index] + reduced))
        return tuple(outputs)


class Joint192ExtensionDetector(StaticJoint192Detector):
    def __init__(self, architecture, seed=42):
        canonical = validate_architecture(architecture)
        # Also protect callers constructing the class directly, not only the factory.
        with torch.random.fork_rng(devices=[]):
            super().__init__(canonical["base_architecture"], seed)
        self.candidate_architecture = canonical
        self.search_space_version = SEARCH_SPACE_VERSION
        changes = canonical["changes"]
        for stage in (1, 2):
            _replace_spatial(self.backbone.stages[stage][0], changes[f"stage{stage}_spatial"],
                             seed, f"backbone.stage{stage}")
        self.neck = ExtensionNeck(self.neck, changes, seed)
        operation = changes["p2_op"]
        reference = self.neck.top_down_blocks["P2"]
        if operation == "identity":
            self.neck.top_down_blocks["P2"] = nn.Identity()
        elif operation in ("mbconv_k3", "mbconv_k5"):
            kernel = int(operation[-1])
            if not (isinstance(reference, MBConvBlock)
                    and reference.dw_conv.kernel_size == (kernel, kernel)):
                block = _new_module(seed, f"neck.P2.{operation}",
                                    lambda: MBConvBlock(48, 48, kernel, stride=1, expand_ratio=3))
                _copy_matching_state(block, reference)
                self.neck.top_down_blocks["P2"] = block
        elif operation in ("sonarop", "frequencylite"):
            _replace_spatial(reference, operation, seed, "neck.P2")
        elif operation in ("dmdc48", "cfem48"):
            cls = DMDCChannelAdapted if operation == "dmdc48" else CFEMChannelAdapted
            self.neck.top_down_blocks["P2"] = _new_module(seed, f"neck.P2.{operation}", cls)
        self._initialization_identity = dict(
            seed=seed, policy=INITIALIZATION_POLICY,
            architecture_sha256=architecture_hash(canonical),
            model_sha256=_hash_named_tensors(self.state_dict().items()),
            head_sha256=_hash_named_tensors(self.head.state_dict().items()),
            lateral_sha256=_hash_named_tensors(self.neck.lateral_convs.state_dict().items()))

    def architecture_contract(self):
        canonical = deepcopy(self.candidate_architecture)
        changes = canonical["changes"]
        base = canonical["base_architecture"]
        return dict(
            search_space_version=SEARCH_SPACE_VERSION, architecture=canonical,
            architecture_id=architecture_id(canonical), architecture_sha256=architecture_hash(canonical),
            base_architecture_id=base_architecture_id(base),
            base_architecture_sha256=base_architecture_hash(base),
            input_chw=[1, 640, 640], input_normalization="PIL_L_bilinear_resize_float32_div255_no_mean_std",
            num_classes=2, levels=list(self.levels), feature_indices=list(self.feature_indices),
            feature_strides=list(self.feature_strides), anchors=deepcopy(self.anchor_sizes),
            neck_channels=48, initialization_policy=INITIALIZATION_POLICY,
            stage_spatial_location="inside_MBConv_preserve_expand_project_overall_stride_and_existing_blur",
            sonarop="BN_DW3_plus_BN_DW5_then_ReLU6_outer_dw_bn_relu_identity",
            frequencylite="fixed_zero_padded_box3_low_plus_DW3_on_x_minus_box3_external_bn_relu_preserved",
            p2_sources={"lateral_only": ["L2"], "adjacent": ["L2", "up_T3"],
                        "direct_p4": ["L2", "up_T3", "up_T4"]}[changes["p2_inputs"]],
            source_semantics="T3_and_T4_are_postblock_top_down_features_before_PAN",
            gated_sum="per_source_spatial_Conv1x1_logits_softmax_across_sources_times_source_count",
            gate_is_input_dependent=changes["p2_fusion"] == "gated_sum",
            gate_is_differentiable=changes["p2_fusion"] == "gated_sum",
            fusion_initialization="sum_equivalent_no_extra_bn_activation",
            dmdc_cfem_scope="channel_only_adaptation_of_local_builder_core_not_new_paper_reproduction",
            dmdc_training_requires_batch_at_least_2=changes["p2_op"] == "dmdc48",
            fixed_lowpass_padding="zeros", prediction_activation="linear",
            board_latency_status="NOT_MEASURED", hardware_admission=False)


def build_model(architecture=None, seed=42):
    return Joint192ExtensionDetector(
        REFERENCE_ARCHITECTURES["E0"] if architecture is None else architecture, seed)


def profile_model(model: nn.Module, input_shape=(1, 1, 640, 640), *, deploy=False):
    """Count executed Conv MACs and known fixed functional convolutions.

    Includes batch size.  Does not estimate latency or count pooling, resize,
    BN, activations, elementwise ops, concatenation, softmax or tensor traffic.
    Profiling/fusion uses a copy and preserves the caller's mode, state and RNG.
    """
    candidate = deepcopy(model).eval()
    if deploy:
        for module in list(candidate.modules()):
            if isinstance(module, ReparamDualRFDepthwise):
                module.switch_to_deploy()
    counts = dict(conv_macs=0, fixed_blur_macs=0, frequencylite_fixed_macs=0)
    hooks = []

    def conv_hook(module, inputs, output):
        counts["conv_macs"] += int(output.numel() * module.kernel_size[0]
                                   * module.kernel_size[1] * module.in_channels // module.groups)

    def frequency_hook(module, inputs, output):
        # Two distinct calls: stride-one smoothing, and strided low output.
        counts["frequencylite_fixed_macs"] += int(9 * (inputs[0].numel() + output.numel()))

    def blur_hook(module, inputs, output):
        batch, _, height, width = output.shape
        counts["fixed_blur_macs"] += int(batch * height * width * module.blur_kernel.shape[0] * 9)

    for module in candidate.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        if isinstance(module, FrequencyLiteDepthwise):
            hooks.append(module.register_forward_hook(frequency_hook))
        if isinstance(module, BlurDownMBConv) and not isinstance(module, DenseDecimateMBConv):
            hooks.append(module.register_forward_hook(blur_hook))
    parameter = next(candidate.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32
    try:
        with torch.no_grad():
            candidate(torch.zeros(tuple(input_shape), device=device, dtype=dtype))
    finally:
        for hook in hooks:
            hook.remove()
    total = sum(counts.values())
    params = sum(parameter.numel() for parameter in candidate.parameters())
    return dict(
        profile_form="deploy_sonarop_folded" if deploy else "training_parameterization_eval_mode",
        input_shape=list(input_shape), params_total=params,
        params_trainable=sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad),
        **counts, functional_macs=counts["fixed_blur_macs"] + counts["frequencylite_fixed_macs"],
        macs_total=total, flops_2x_macs=2 * total,
        count_scope="executed_Conv2d_plus_fixed_blur_and_FrequencyLite_functional_convolutions_includes_batch",
        excluded_operations=["pooling", "interpolation", "batch_norm", "activations", "elementwise",
                             "concatenation", "softmax", "memory_traffic"],
        latency_measured=False, hardware_admission=False)


__all__ = ["SEARCH_SPACE_VERSION", "INITIALIZATION_POLICY", "DEFAULT_CHANGES",
           "REFERENCE_ARCHITECTURES", "validate_architecture", "architecture_id",
           "architecture_hash", "DMDCChannelAdapted", "CFEMChannelAdapted",
           "P2SourceFusion", "ExtensionNeck", "Joint192ExtensionDetector",
           "build_model", "profile_model"]
