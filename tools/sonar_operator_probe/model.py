"""Isolated operator probes on the paired E0 + C01b detection protocol.

These software candidates do not extend the 192-space or claim hardware
admission. Existing backbones, head, loss and historical runs are unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.models.sonar_op import ReparamDualRFDepthwise
from hwnas_fpga.models.sonar_operator_pool_p0 import make_spatial_operator
from tools.detector_elastic_pilot.model import ElasticFourScaleNeck, _hash_named_tensors, _pointwise
from tools.sonar_head_c01b.model import (
    C01bDetector, DEFAULT_SOURCE_ROOT, SEEDS, batch_loss, build_criterion,
    criterion_metadata, decode, loss_from_output,
)
from tools.sonar_joint192_validation.model import BlurDownMBConv, DenseDecimateMBConv
from tools.sonar_joint192_extension_20261013.model import (
    CFEMChannelAdapted, DMDCChannelAdapted, P2SourceFusion,
)

INITIALIZATION_POLICY = "OPERATOR_PROBE_E0_C01B_SHARED_TENSORS_SEMANTIC_CPU_SEED_V1"
VARIANTS = (
    "control", "dw3_stage7", "dw3_stage2",
    "sonarop_stage7", "sonarop_weighted_stage7", "sonarop_stage2",
    "dilated_dw3_stage2", "directional_dw_stage5", "dcn_stage1",
    "frequency_lite_stage1", "nagr_p2", "nagr_p2_ungated",
    "dmdc_p2", "cfem_p2",
    "p2_lateral_only", "p2_extra_p4", "p2_concat", "p2_gated",
    "p3_weighted", "dense_decimate_stage1", "blur_stage1",
)
ALIAS_VARIANTS = ("dw5_stage7",)
REUSED_EXTENSION_PATH = Path(__file__).resolve().parents[1] / "sonar_joint192_extension_20261013/model.py"
REUSED_EXTENSION_SHA256 = hashlib.sha256(REUSED_EXTENSION_PATH.read_bytes()).hexdigest()


def allowed_change_prefixes(name: str) -> tuple[str, ...]:
    """Independent location contract, never inferred from observed differences."""
    if name in ("control", "dw5_stage7"):
        return ()
    if name.startswith(("nagr_p2",)):
        return ("neck.p2_post.",)
    if name in ("dmdc_p2", "cfem_p2"):
        return ("neck.top_down_blocks.P2.",)
    if name.startswith("p2_"):
        return ("neck.p2_fusion.",)
    if name == "p3_weighted":
        return ("neck.alpha",)
    if name in ("dense_decimate_stage1", "blur_stage1"):
        return ("backbone.stages.1.0.dw_conv.", "backbone.stages.1.0.blur_kernel")
    if name in VARIANTS and "stage" in name:
        stage = int(name.rsplit("stage", 1)[-1])
        prefix = f"backbone.stages.{stage}.0."
        return (prefix + "dw_conv.", prefix + "dw_bn.") if name.startswith("sonarop_") else (prefix + "dw_conv.",)
    raise ValueError(f"Unknown scope contract: {name!r}")


def _verify_change_scope(control, actual, prefixes):
    changed = sorted(name for name, old in control.items()
                     if name not in actual or old.shape != actual[name].shape or not torch.equal(old, actual[name]))
    added = sorted(set(actual) - set(control))
    unexpected = [name for name in (*changed, *added) if not name.startswith(prefixes)]
    if unexpected:
        raise ValueError(f"Probe changed tensors outside its registered location: {unexpected}")
    return changed, added


def variant_spec(name: str) -> dict[str, Any]:
    if name not in (*VARIANTS, *ALIAS_VARIANTS):
        raise ValueError(f"Unknown operator probe variant: {name!r}")
    definitions = {
        "control": ("none", "unchanged E0 + C01b", [], "reference", "no change"),
        "dw5_stage7": ("backbone.stages[7][0].dw_conv", "native DW5 unchanged", ["control"], "exact_alias", "no new parameter"),
        "dw3_stage7": ("backbone.stages[7][0].dw_conv", "DW3", ["control"], "candidate_effect", "5x5 to 3x3 spatial weight shape"),
        "dw3_stage2": ("backbone.stages[2][0].dw_conv", "DW3", ["control"], "candidate_effect", "5x5 to 3x3 spatial weight shape"),
        "sonarop_stage7": ("backbone.stages[7][0].dw_conv", "ReLU6(BN(DW3)+BN(DW5))", ["control", "dw3_stage7"], "candidate_effect", "two DW branches and branch BN; outer DW BN/ReLU removed"),
        "sonarop_weighted_stage7": ("backbone.stages[7][0].dw_conv", "ReLU6(BN(DW3)+alpha*BN(DW5)), alpha>=0 initial1", ["sonarop_stage7", "control"], "paired_scalar_ablation", "one global scalar alpha; branch tensors paired to sonarop_stage7"),
        "sonarop_stage2": ("backbone.stages[2][0].dw_conv", "ReLU6(BN(DW3)+BN(DW5))", ["control", "dw3_stage2"], "candidate_effect", "two DW branches and branch BN; earlier placement"),
        "dilated_dw3_stage2": ("backbone.stages[2][0].dw_conv", "DW3 dilation2 padding2", ["dw3_stage2", "control"], "candidate_effect", "3x3 sparse weights, effective5x5; DW3 weights paired to dw3_stage2"),
        "directional_dw_stage5": ("backbone.stages[5][0].dw_conv", "0.5*(DW1x5+DW5x1)", ["control"], "candidate_effect", "two directional branches replace native DW5"),
        "dcn_stage1": ("backbone.stages[1][0].dw_conv", "P0 per-channel offset depthwise DeformConv3", ["control"], "candidate_effect", "offset branch zero initialized; native DW3 weights copied"),
        "frequency_lite_stage1": ("backbone.stages[1][0].dw_conv", "P0 fixed avg3 low + DW3(x-avg3(x))", ["control"], "candidate_effect", "learned high DW3 paired to native DW3; zero padding"),
        "nagr_p2": ("neck post-P2 top_down MBConv", "x+sigmoid(PW(local channel variance))*F(x)", ["nagr_p2_ungated", "control"], "candidate_effect", "F=PW48-DW3-PW48 with BN, no expansion; gate48->1 initweight0 bias-3"),
        "nagr_p2_ungated": ("neck post-P2 top_down MBConv", "x+F(x)", ["control"], "candidate_effect", "same residual tensors as nagr_p2; no gate"),
        "dmdc_p2": ("neck P2 top_down block", "reused DMDCChannelAdapted48 replaces MBConv", ["control"], "candidate_effect", "48-channel adaptation of local paper-derived DMDC; all weights trainable; no paper reproduction or native32 HLS transfer"),
        "cfem_p2": ("neck P2 top_down block", "reused CFEMChannelAdapted48 replaces MBConv", ["control"], "candidate_effect", "48-channel adaptation of local paper-derived CFEM; all weights trainable; no paper reproduction or native32 HLS transfer"),
        "p2_lateral_only": ("neck P2 pre-MBConv fusion", "L2 only, retain P2 MBConv", ["control"], "connection_ablation", "no new parameter"),
        "p2_extra_p4": ("neck P2 pre-MBConv fusion", "L2+up(T3)+up(T4)", ["control"], "connection_ablation", "no new parameter; T3 already contains T4, correlated extra path"),
        "p2_concat": ("neck P2 pre-MBConv fusion", "linear PW1x1(concat(L2,up(T3)))", ["control"], "candidate_effect", "96->48 linear projection initial[I,I] reproduces sum; trainable capacity added"),
        "p2_gated": ("neck P2 pre-MBConv fusion", "reused P2SourceFusion input-dependent spatial Conv1 gates, softmax*2", ["control"], "candidate_effect", "two 48->1 Conv1 logits initial0, normalized nonnegative; initial weights1 preserve sum scale; not global scalar gates"),
        "p3_weighted": ("neck P3 pre-MBConv fusion", "L3+alpha*up(T4), alpha>=0 initial1", ["control"], "paired_scalar_ablation", "one global scalar; modified T3 propagates to T2"),
        "dense_decimate_stage1": ("backbone.stages[1][0]", "dense DW-BN-ReLU6, ::2 decimation, projection", ["control", "blur_stage1"], "downsample_attribution", "same trainable tensors as control and blur_stage1; no smoothing"),
        "blur_stage1": ("backbone.stages[1][0]", "dense DW-BN-ReLU6, triangle3 lowpass stride2, projection", ["control", "dense_decimate_stage1"], "positive_reference_retest", "same trainable tensors as control; fixed zero-padded triangle kernel"),
    }
    position, formula, parents, effect_scope, change = definitions[name]
    return dict(
        variant=name, position=position, formula=formula, parent_comparators=parents,
        effect_scope=effect_scope, new_parameters=change,
        control="E0_B1_stage1k3_stage2k5_stage5k5_stage7k5_C01b6_direct_GIoU",
        fixed_head="C01b four native E0 levels, raw6, noDFL, GIoU, TAL",
        seed_policy=INITIALIZATION_POLICY, board_latency_status="NOT_MEASURED",
        test_evaluated=False, alias_of="control" if name == "dw5_stage7" else None,
        allowed_change_prefixes=list(allowed_change_prefixes(name)),
    )


def _seed(seed: int, component: str) -> None:
    digest = hashlib.sha256(f"{INITIALIZATION_POLICY}:{seed}:{component}".encode()).digest()
    torch.random.default_generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))


def _construct(seed: int, component: str, factory):
    with torch.random.fork_rng(devices=[]):
        _seed(seed, component)
        return factory()


def _copy_matching(reference: nn.Module, candidate: nn.Module) -> None:
    old = reference.state_dict()
    candidate.load_state_dict({name: old[name] if name in old and old[name].shape == tensor.shape
                               else tensor for name, tensor in candidate.state_dict().items()}, strict=True)


def _dw3(reference: nn.Conv2d, seed: int, stage: int) -> nn.Conv2d:
    if reference.kernel_size == (3, 3):
        return deepcopy(reference)
    return _construct(seed, f"stage{stage}.dw3", lambda: nn.Conv2d(
        reference.in_channels, reference.out_channels, 3, stride=reference.stride,
        padding=1, groups=reference.groups, bias=False))


class WeightedDualRFDepthwise(ReparamDualRFDepthwise):
    """One direct global nonnegative scalar; initial forward equals the original."""
    def __init__(self, channels: int, stride: int = 1):
        super().__init__(channels, stride)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, value):
        if self.deploy:
            return super().forward(value)
        return self.activation(self.branch3(value) + self.alpha.clamp_min(0) * self.branch5(value))

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return super().get_equivalent_kernel_bias()
        if self.training:
            raise RuntimeError("fusion requires eval mode with frozen BN running statistics")
        weight3, bias3 = self._fold_conv_bn(self.branch3[0], self.branch3[1])
        weight5, bias5 = self._fold_conv_bn(self.branch5[0], self.branch5[1])
        alpha = self.alpha.clamp_min(0)
        return F.pad(weight3, (1, 1, 1, 1)) + alpha * weight5, bias3 + alpha * bias5


class LocalResidual(nn.Module):
    def __init__(self, channels: int = 48):
        super().__init__()
        self.pw_in = _pointwise(channels, channels)
        self.dw = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                nn.BatchNorm2d(channels), nn.ReLU6(inplace=True))
        self.pw_out = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, value):
        return self.pw_out(self.dw(self.pw_in(value)))


class NAGR(nn.Module):
    """Local-variance gating is a hypothesis; no noise labels are inferred here."""
    def __init__(self, seed: int, gated: bool):
        super().__init__()
        self.residual = _construct(seed, "nagr.residual", LocalResidual)
        self.gated = gated
        if gated:
            self.gate = _construct(seed, "nagr.gate", lambda: nn.Conv2d(48, 1, 1, bias=True))
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(-3.)

    def gate_map(self, value):
        local_mean = F.avg_pool2d(value, 3, 1, 1)
        variance = (F.avg_pool2d(value.square(), 3, 1, 1) - local_mean.square()).clamp_min(0)
        return torch.sigmoid(self.gate(variance))

    def forward(self, value):
        residual = self.residual(value)
        return value + (self.gate_map(value) * residual if self.gated else residual)


class ProbeNeck(ElasticFourScaleNeck):
    """Reuse registered E0 modules under the same state keys; no hooks or PAN."""
    def __init__(self, original: ElasticFourScaleNeck, seed: int, variant: str):
        nn.Module.__init__(self)
        if original.bottom_up_pan or original.out_channels != 48:
            raise ValueError("Operator probe requires native E0 U48 top-down neck")
        self.in_channels, self.out_channels = original.in_channels, original.out_channels
        self.levels, self.bottom_up_pan = original.levels, original.bottom_up_pan
        self.lateral_convs, self.top_down_blocks = original.lateral_convs, original.top_down_blocks
        self.bottom_up_projections = original.bottom_up_projections
        self.bottom_up_blocks = original.bottom_up_blocks
        self.variant = variant
        if variant in ("nagr_p2", "nagr_p2_ungated"):
            self.p2_post = NAGR(seed, variant == "nagr_p2")
        else:
            self.p2_post = nn.Identity()
        if variant in ("dmdc_p2", "cfem_p2"):
            cls = DMDCChannelAdapted if variant == "dmdc_p2" else CFEMChannelAdapted
            self.top_down_blocks["P2"] = _construct(seed, f"neck.{variant}", lambda: cls(48))
        count = 1 if variant == "p2_lateral_only" else 3 if variant == "p2_extra_p4" else 2
        mode = {"p2_concat": "concat_project", "p2_gated": "gated_sum"}.get(variant, "sum")
        self.p2_fusion = _construct(seed, f"p2.fusion.{mode}.{count}", lambda: P2SourceFusion(48, count, mode))
        if variant == "p3_weighted":
            self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, features):
        if len(features) != 4:
            raise ValueError("Operator probe requires all four E0 feature maps")
        lateral = [self.lateral_convs[level](feature) for level, feature in zip(self.levels, features)]
        fused = list(lateral)
        for index in range(2, -1, -1):
            shallow = lateral[index]
            deep = F.interpolate(fused[index + 1], size=shallow.shape[-2:], mode="nearest")
            if index == 0 and self.variant == "p2_lateral_only":
                merged = self.p2_fusion([shallow])
            elif index == 0 and self.variant == "p2_extra_p4":
                merged = self.p2_fusion([shallow, deep, F.interpolate(fused[2], size=shallow.shape[-2:], mode="nearest")])
            elif index == 0:
                merged = self.p2_fusion([shallow, deep])
            elif index == 1 and self.variant == "p3_weighted":
                merged = shallow + self.alpha.clamp_min(0) * deep
            else:
                merged = shallow + deep
            fused[index] = self.top_down_blocks[self.levels[index]](merged)
        fused[0] = self.p2_post(fused[0])
        return tuple(fused)


def _apply_backbone_variant(model, name: str, seed: int):
    if name in ("control", "dw5_stage7") or "_p2" in name or name.startswith("p2_") or name == "p3_weighted":
        return
    stage = int(name.rsplit("stage", 1)[-1])
    block = model.backbone.stages[stage][0]
    reference = block.dw_conv
    if not isinstance(reference, nn.Conv2d):
        raise ValueError("Expected unmodified native depthwise convolution")
    if name.startswith("dw3_"):
        block.dw_conv = _dw3(reference, seed, stage)
    elif name.startswith("sonarop_"):
        constructor = WeightedDualRFDepthwise if name == "sonarop_weighted_stage7" else ReparamDualRFDepthwise
        operator = _construct(seed, f"stage{stage}.sonarop", lambda: constructor(reference.in_channels, reference.stride[0]))
        with torch.no_grad():
            operator.branch3[0].weight.copy_(_dw3(reference, seed, stage).weight)
            if reference.kernel_size != (5, 5):
                raise ValueError("SonarOp pairing expects native E0 DW5")
            operator.branch5[0].weight.copy_(reference.weight)
        operator.branch3[1].load_state_dict(block.dw_bn.state_dict())
        operator.branch5[1].load_state_dict(block.dw_bn.state_dict())
        block.dw_conv, block.dw_bn, block.dw_relu = operator, nn.Identity(), nn.Identity()
    elif name in ("dense_decimate_stage1", "blur_stage1"):
        cls = DenseDecimateMBConv if name.startswith("dense") else BlurDownMBConv
        if reference.stride != (2, 2):
            raise ValueError("Downsample attribution requires native stage1 stride2")
        candidate = _construct(seed, "stage1.dense_downsample", lambda: cls(
            block.expand_conv.in_channels, block.project_conv.out_channels,
            reference.kernel_size[0], expand_ratio=reference.in_channels // block.expand_conv.in_channels))
        _copy_matching(block, candidate)
        model.backbone.stages[stage][0] = candidate
    else:
        operation = {"dilated_dw3_stage2": "DilatedDW3", "directional_dw_stage5": "DirectionalDW",
                     "dcn_stage1": "DCN", "frequency_lite_stage1": "FrequencyLite"}[name]
        operator = _construct(seed, f"stage{stage}.{operation}", lambda: make_spatial_operator(operation, reference))
        with torch.no_grad():
            if operation == "DilatedDW3":
                operator.weight.copy_(_dw3(reference, seed, stage).weight)
            elif operation == "DCN":
                operator.deform.weight.copy_(reference.weight)
            elif operation == "FrequencyLite":
                operator.high_dw.weight.copy_(reference.weight)
        block.dw_conv = operator


class OperatorProbeDetector(C01bDetector):
    def __init__(self, variant: str, seed: int, source_root: Path):
        super().__init__(seed, source_root)
        self.variant = variant
        control_state = {name: tensor.clone() for name, tensor in self.state_dict().items()}
        control_identity = deepcopy(self._initialization)
        _apply_backbone_variant(self, variant, seed)
        if variant in ("nagr_p2", "nagr_p2_ungated", "dmdc_p2", "cfem_p2",
                       "p2_lateral_only", "p2_extra_p4", "p2_concat", "p2_gated", "p3_weighted"):
            self.neck = ProbeNeck(self.neck, seed, variant)
        actual = self.state_dict()
        allowed = allowed_change_prefixes(variant)
        changed, added = _verify_change_scope(control_state, actual, allowed)
        # Protect all tensor names outside the independently specified location.
        # Equal numerical values at changed semantic roles are not identity proof.
        shared_names = sorted(name for name in control_state if not name.startswith(allowed))
        shared_before = _hash_named_tensors((name, control_state[name]) for name in shared_names)
        shared_after = _hash_named_tensors((name, actual[name]) for name in shared_names)
        if shared_before != shared_after:
            raise ValueError("An unchanged paired tensor drifted during probe construction")
        self._initialization.update(
            policy=INITIALIZATION_POLICY, variant=variant,
            control_initialization=control_identity,
            backbone_sha256=_hash_named_tensors(self.backbone.state_dict().items()),
            neck_sha256=_hash_named_tensors(self.neck.state_dict().items()),
            head_sha256=_hash_named_tensors(self.head.state_dict().items()),
            model_sha256=_hash_named_tensors(actual.items()),
            unchanged_tensor_sha256=shared_after, control_unchanged_tensor_sha256=shared_before,
            shared_tensor_sha256={name: _hash_named_tensors(((name, actual[name]),)) for name in shared_names},
            unchanged_tensor_count=len(shared_names), changed_or_removed_control_tensors=changed,
            allowed_change_prefixes=list(allowed), newly_named_tensors=added,
        )
        if self._initialization["head_sha256"] != control_identity["head_sha256"]:
            raise ValueError("Operator probe changed C01b head initialization")

    def metadata(self):
        result = super().metadata()
        result.update(adapter="operator_probe_E0_C01b", operator_probe=variant_spec(self.variant),
                      adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      initialization=self.initialization_identity(),
                      reused_extension_source_sha256=REUSED_EXTENSION_SHA256,
                      context_scope="channel adaptation of local literature modules; no new original-paper reproduction or FPGA evidence",
                      training_requires_batch_at_least_2=self.variant == "dmdc_p2",
                      backbone_reference="complete E0 B1, only registered probe location changed")
        return result


def build_model(variant: str, seed: int, source_root: str | Path | None = None) -> OperatorProbeDetector:
    variant_spec(variant)
    if hashlib.sha256(REUSED_EXTENSION_PATH.read_bytes()).hexdigest() != REUSED_EXTENSION_SHA256:
        raise ValueError("Reused extension source changed after module import; freeze and restart the process")
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Operator probe seed must be integer42,43,44")
    with torch.random.fork_rng(devices=[]):
        return OperatorProbeDetector(variant, seed, Path(source_root or DEFAULT_SOURCE_ROOT).resolve())


def parameter_groups(model: OperatorProbeDetector, weight_decay: float = 1e-4):
    """Same ordinary AdamW decay; alpha/gate scalars have no weight decay."""
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    special, ordinary = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (special if name.endswith(".alpha") else ordinary).append(parameter)
    groups = [{"name": "ordinary_model", "params": ordinary, "weight_decay": weight_decay}]
    if special:
        groups.append({"name": "fusion_scalars", "params": special, "weight_decay": 0.0})
    return groups


@torch.no_grad()
def project_nonnegative_(model: OperatorProbeDetector) -> None:
    for name, parameter in model.named_parameters():
        if name.endswith(".alpha"):
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"Nonfinite operator probe scalar {name}")
            parameter.clamp_(min=0)


__all__ = ["VARIANTS", "ALIAS_VARIANTS", "INITIALIZATION_POLICY", "variant_spec", "allowed_change_prefixes", "build_model", "OperatorProbeDetector",
           "ProbeNeck", "NAGR", "WeightedDualRFDepthwise", "parameter_groups",
           "project_nonnegative_", "build_criterion", "criterion_metadata", "batch_loss", "loss_from_output", "decode"]
