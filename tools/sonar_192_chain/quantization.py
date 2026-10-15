"""Checkpoint- and train-source-bound W8A8 PTQ for the complete 192v2 raw DAG.

The runtime reference uses NumPy integer sums/requantization, including grouped
convolutions. An independent torch FP64 fake reference computes SiLU from its
formula rather than reading the integer LUT. This is software evidence only.
Unsupported precision profiles are rejected before calibration or export.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace
from hwnas_fpga.deploy.h2_ptq import exponent_from_maxabs
from tools.fpga_h2 import numeric as h2
from tools.modnas_comparison.quantization import TrainCalibration, model_state_hash
from .contract import digest, sha256_file, validate_identity
from . import graph as graph_module
from .graph import LEVELS, build_graph, float_graph, raw_predictions

SUPPORTED_PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")
NUMERIC_CONTRACT = {
    "version": "sonar192v2_w8a8_group_silu_lut_v1", "activation_bits": 8,
    "weight_bits": 8, "accumulator_bits": 32, "requant_product_bits": 64,
    "range": [-128, 127], "zero_point": 0, "rounding": "nearest_even",
    "overflow": "twos_complement_modulo_2pow32", "scale": "positive_power_of_two",
    "scale_exponent_range": [-24, 16], "weight_granularity": "per_tensor",
    "conv": "INT64_exact_sum_including_bias_then_INT32_wrap_then_requant_and_activation",
    "groups": "all_divisors_of_Cin_and_Cout_explicit_OIHW_group_order",
    "bias": "nearest_even_FP64_to_saturated_INT32_at_input_times_weight_scale",
    "weight": "nearest_even_FP64_to_saturated_INT8_at_weight_scale",
    "add": "align_each_input_to_max_exponent_INT8_then_INT32_sum_requant",
    "blur": "fixed_triangle3_numerators_zero_pad_stride2_INT32_sum_nearest_even_div16_same_scale",
    "concat": "align_each_branch_to_max_exponent_INT8_then_channel_concat_ltrb4_class2",
    "relu": "integer_clamp_min_zero_same_scale",
    "relu6": "after_conv_requant_clamp_0_nearest_even_6_div_output_scale",
    "silu": "LUT256_signed_index_q_plus128_nearest_even_clip_SiLU(q*input_scale)/output_scale",
    "silu_lut_generation": "python_math_exp_stable_float64_offline_table_bytes_are_authoritative",
    "silu_independent_reference": "torch_float64_SiLU_formula_then_nearest_even_not_LUT_lookup",
    "output": "four_raw_heads_6_channels_ltrb_relu4_class_quality_logits2",
    "host_postprocessing": "decode_sigmoid_NMS_separate_not_integer_graph",
}
NUMERIC_CONTRACT_HASH = digest(NUMERIC_CONTRACT)


def numeric_contract(profile="W8A8"):
    if profile not in SUPPORTED_PROFILES:
        raise NotImplementedError("unsupported 192v2 precision profile: " + str(profile))
    if profile == "W8A8":
        return deepcopy(NUMERIC_CONTRACT), NUMERIC_CONTRACT_HASH
    value = dict(version="sonar192v2_body8_head16_group_silu_lut_v1", profile=profile,
                 body=deepcopy(NUMERIC_CONTRACT), body_contract_sha256=NUMERIC_CONTRACT_HASH,
                 body_activation_bits=8, body_weight_bits=8, head_input_bits=16,
                 head_activation_bits=16, head_weight_bits=16 if profile == "HEAD_W16A16" else 8,
                 head_range=[-32768,32767], accumulator_bits=32, requant_product_bits=64,
                 rounding="nearest_even", overflow="twos_complement_modulo_2pow32", zero_point=0,
                 scale="positive_power_of_two", scale_exponent_range=[-24,16], weight_granularity="per_tensor",
                 boundary="exactly_four_shared_INT8_neck_to_INT16_head_integer_requantize_nodes",
                 conv="INT64_exact_sum_including_bias_then_INT32_wrap_then_INT16_requant",
                 weight="nearest_even_FP64_to_saturated_signed_INT8_or_INT16_at_weight_scale",
                 bias="nearest_even_FP64_to_saturated_INT32_at_input_times_weight_scale",
                 silu="LUT65536_signed_index_q_plus32768_nearest_even_clip_SiLU(q*input_scale)/output_scale",
                 silu_lut_generation="python_math_exp_stable_float64_offline_table_bytes_are_authoritative",
                 silu_independent_reference="torch_float64_SiLU_formula_then_nearest_even_not_LUT_lookup",
                 relu="integer_clamp_min_zero_same_scale",
                 concat="align_both_INT16_branches_to_max_exponent_then_concat_ltrb4_class2",
                 output="four_raw_INT16_heads_6_channels_ltrb_relu4_class_quality_logits2",
                 host_postprocessing="decode_sigmoid_NMS_separate_not_integer_graph")
    return value, digest(value)


def _bounds(bits):
    if bits not in (8,16):
        raise ValueError("only signed INT8/INT16 tensors are registered")
    return -(1 << (bits-1)), (1 << (bits-1))-1


def _dtype(bits):
    _bounds(bits)
    return np.dtype("i1" if bits == 8 else "<i2")


def calibration_from_training_manifest(dataset_root, manifest_path, *, num_images=64, formal=True):
    """Bind calibration to the same execution manifest as the selected EMA run."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    train = manifest["inputs"]["datasets"]["train"]
    if manifest.get("test_evaluated") is not False or digest(train["entries"]) != train["sha256"]:
        raise ValueError("training execution manifest train inventory or sealed-test status differs")
    calibration = TrainCalibration(dataset_root, train["entries"], num_images=num_images, formal=formal)
    calibration.manifest.pop("calibration_sha256")
    calibration.manifest.update(training_execution_manifest_path=str(path),
                                training_execution_manifest_sha256=sha256_file(path),
                                frozen_training_images=train["images"],
                                training_source_verified=True)
    calibration.manifest["calibration_sha256"] = digest(calibration.manifest)
    calibration.verify_manifest()
    return calibration


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_sha(value):
    return array_sha(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())


def _quantize(value, exponent, *, bits=8, bias=False):
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite quantization input")
    lo, hi = (-(1 << 31), (1 << 31)-1) if bias else _bounds(bits)
    return np.clip(np.rint(value / (2.0**exponent)), lo, hi).astype("<i4" if bias else _dtype(bits))


def silu_lut(input_exp, output_exp, *, input_bits=8, output_bits=8):
    """The complete signed domain, frozen with a byte hash in every export."""
    low, high = _bounds(output_bits)
    input_low, input_high = _bounds(input_bits)
    values = []
    for q in range(input_low, input_high+1):
        x = q * (2.0**input_exp)
        y = x / (1.0 + math.exp(-x)) if x >= 0 else x * math.exp(x) / (1.0 + math.exp(x))
        values.append(max(low, min(high, round(y / (2.0**output_exp)))))
    return np.asarray(values, dtype=_dtype(output_bits))


def grouped_conv_acc(x, weight, bias, *, stride, padding, groups):
    """CHW/OIHW integer convolution; no float matmul, torch, or fake runtime."""
    x, weight, bias = np.asarray(x), np.asarray(weight), np.asarray(bias)
    if x.dtype not in (np.int8,np.int16) or weight.dtype not in (np.int8,np.int16) or bias.dtype != np.int32:
        raise ValueError("integer conv requires signed INT8/INT16 input/weight and INT32 bias")
    ci, h, w = x.shape
    co, per_group, kh, kw = weight.shape
    if (groups < 1 or ci % groups or co % groups or per_group != ci//groups
            or kh != kw or bias.shape != (co,)):
        raise ValueError("group convolution geometry mismatch")
    if groups in (1, ci) and (groups == 1 or co == ci):
        return h2.conv_acc(x, weight, bias, stride, padding, groups)
    result = []
    for group in range(groups):
        part = h2.conv_acc(x[group*(ci//groups):(group+1)*(ci//groups)],
                           weight[group*(co//groups):(group+1)*(co//groups)],
                           bias[group*(co//groups):(group+1)*(co//groups)], stride, padding, 1)
        result.append(part)
    return np.concatenate(result, axis=0)


def _int_requant(values, source_exp, output_exp, *, bits=8):
    multiplier,shift = h2.ratio(source_exp, output_exp)
    return _int_multiplier(values,multiplier,shift,bits=bits)


def _int_multiplier(values,multiplier,shift,*,bits=8):
    array = np.asarray(values,dtype=np.int64)
    if np.any(array < -(2**31)) or np.any(array > 2**31-1):
        raise ValueError("requantization accumulator must first wrap to INT32")
    if not 1 <= int(multiplier) <= 2**31-1:
        raise ValueError("requantization multiplier outside contract")
    return np.clip(h2.round_shift(array*int(multiplier),shift),*_bounds(bits)).astype(_dtype(bits))


def _float_requant(values, source_exp, output_exp, *, bits=8):
    # All values are integers represented exactly in FP64. Power-of-two ratios
    # retain exact ties even when the unclipped product exceeds 2**53.
    h2.ratio(source_exp, output_exp)
    return (values * (2.0**(source_exp-output_exp))).round().clamp(*_bounds(bits))


class IntegerReference:
    execution_kind = "NUMPY_INTEGER_INT64_SUM_INT32_WRAP_INT8_OR_INT16_WITH_EXPORTED_SILU_LUT"

    def __init__(self, graph, parameters, luts):
        self.graph = deepcopy(graph)
        self.parameters = {name: {key: value.copy() for key, value in row.items()}
                           for name, row in parameters.items()}
        self.luts = {name: value.copy() for name, value in luts.items()}

    def parameter_hash(self):
        rows = {name: {kind + "_sha256": array_sha(value) for kind, value in row.items()}
                for name, row in sorted(self.parameters.items())}
        return digest(dict(parameters=rows, luts={name: array_sha(value) for name, value in sorted(self.luts.items())}))

    def quantize_input(self, images):
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if np.asarray(images).ndim != 4 or list(np.asarray(images).shape[1:]) != self.graph["input_shape"]:
            raise ValueError("integer input geometry mismatch")
        return _quantize(images, self.graph["input_exp"])

    def forward(self, images, *, retain=True):
        images = np.asarray(images)
        if images.dtype != np.int8 or images.ndim != 4 or list(images.shape[1:]) != self.graph["input_shape"]:
            raise ValueError("integer runtime consumes already-quantized INT8 NCHW")
        samples = []
        for image in images:
            values = {"image": image}
            for node in self.graph["nodes"]:
                src = [values[name] for name in node["inputs"]]
                op, name, exps, eo = node["op"], node["id"], node["input_exps"], node["output_exp"]
                if op == "conv":
                    row = self.parameters[name]
                    acc = grouped_conv_acc(src[0], row["weight"], row["bias"], stride=node["stride"],
                                           padding=node["padding"], groups=node["groups"])
                    value = _int_multiplier(acc,node["multiplier"],node["shift"],bits=node.get("output_bits",8))
                    if node["activation"] == "relu":
                        value = np.maximum(value,0).astype(value.dtype)
                    elif node["activation"] == "relu6":
                        bound = min(_bounds(node.get("output_bits",8))[1], max(0,round(6.0/(2.0**eo))))
                        value = np.clip(value,0,bound).astype(value.dtype)
                    elif node["activation"] != "linear":
                        raise ValueError("unsupported convolution activation")
                elif op in ("add", "residual_add"):
                    value = h2.aligned_add(src[0], src[1], exps[0], exps[1], eo)
                elif op == "blur":
                    c = src[0].shape[0]
                    weights = np.asarray(node["fixed_kernel"], dtype=np.int8)[None, None].repeat(c, axis=0)
                    acc = grouped_conv_acc(src[0], weights, np.zeros(c, dtype=np.int32), stride=2, padding=1, groups=c)
                    value = np.clip(h2.round_shift(acc, 4), -128, 127).astype(np.int8)
                elif op == "nearest":
                    value = np.repeat(np.repeat(src[0], 2, axis=1), 2, axis=2)
                elif op == "maxpool":
                    c, h, w = src[0].shape
                    value = src[0].reshape(c, h//2, 2, w//2, 2).max(axis=(2, 4))
                elif op == "silu":
                    value = self.luts[name][src[0].astype(np.int32)+(1 << (node.get("input_bits",8)-1))]
                elif op == "relu":
                    value = np.maximum(src[0], 0).astype(_dtype(node.get("output_bits",8)))
                elif op == "concat":
                    value = np.concatenate([_int_requant(x.astype(np.int32), exp, eo,bits=node.get("output_bits",8)) for x, exp in zip(src, exps)], axis=0)
                elif op == "requantize":
                    value = _int_requant(src[0].astype(np.int32),exps[0],eo,bits=node["output_bits"])
                else:
                    raise ValueError("unsupported integer op: " + op)
                if value.dtype != _dtype(node.get("output_bits",8)) or list(value.shape) != node["output_shape"]:
                    raise ValueError("integer node dtype/shape mismatch: " + name)
                values[name] = value
            samples.append(values if retain else {name: values[name] for name in self.graph["outputs"].values()})
        return {name: np.stack([sample[name] for sample in samples]) for name in samples[0]}

    def parameter_package(self):
        contract, checksum = numeric_contract(self.graph.get("precision_profile","W8A8"))
        return dict(graph=deepcopy(self.graph), numeric_contract=contract, numeric_contract_hash=checksum,
                    weights={name: torch.from_numpy(row["weight"].copy()) for name, row in self.parameters.items()},
                    biases={name: torch.from_numpy(row["bias"].copy()) for name, row in self.parameters.items()},
                    luts={name: torch.from_numpy(table.copy()) for name, table in self.luts.items()},
                    execution_kind=self.execution_kind, hardware_verified=False)


class FakeReference(nn.Module):
    execution_kind = "TORCH_FP64_EXACT_INTEGER_VALUED_FAKE_WITH_FORMULA_SILU_NO_LUT"

    def __init__(self, graph, parameters):
        super().__init__()
        self.graph, self.keys = deepcopy(graph), {}
        for index, (name, row) in enumerate(parameters.items()):
            self.keys[name] = index
            self.register_buffer(f"weight_{index}", torch.from_numpy(row["weight"].copy()).double())
            self.register_buffer(f"bias_{index}", torch.from_numpy(row["bias"].copy()).double())

    def forward_all(self, images):
        if images.ndim != 4 or list(images.shape[1:]) != self.graph["input_shape"]:
            raise ValueError("fake input geometry mismatch")
        values = {"image": (images.double()/(2.0**self.graph["input_exp"])).round().clamp(-128,127)}
        for node in self.graph["nodes"]:
            src = [values[name] for name in node["inputs"]]
            op, name, exps, eo = node["op"], node["id"], node["input_exps"], node["output_exp"]
            bits = node.get("output_bits",8)
            if op == "conv":
                index = self.keys[name]
                weight, bias = getattr(self, f"weight_{index}"), getattr(self, f"bias_{index}")
                if math.prod(weight.shape[1:])*(1 << (node.get("input_bits",8)-1))*(1 << (node.get("weight_bits",8)-1)) + 2**31 >= 2**53:
                    raise ValueError("fake exact-integer accumulation bound exceeded")
                value = F.conv2d(src[0], weight, bias, stride=node["stride"], padding=node["padding"], groups=node["groups"])
                value = torch.remainder(value + 2**31, 2**32) - 2**31
                value = _float_requant(value, exps[0]+node["weight_exp"], eo,bits=bits)
                if node["activation"] == "relu":
                    value = value.clamp_min(0)
                elif node["activation"] == "relu6":
                    value = value.clamp(0, min(_bounds(bits)[1], max(0, round(6.0/(2.0**eo)))))
                elif node["activation"] != "linear":
                    raise ValueError("unsupported activation")
            elif op in ("add", "residual_add"):
                common = max(exps)
                value = _float_requant(sum(_float_requant(x, exp, common) for x, exp in zip(src, exps)), common, eo)
            elif op == "blur":
                weights = torch.tensor(node["fixed_kernel"], dtype=torch.float64, device=src[0].device)[None,None].repeat(src[0].shape[1],1,1,1)
                value = (F.conv2d(src[0], weights, stride=2, padding=1, groups=src[0].shape[1])/16).round().clamp(-128,127)
            elif op == "nearest":
                value = F.interpolate(src[0], scale_factor=2, mode="nearest")
            elif op == "maxpool":
                value = F.max_pool2d(src[0], 2, 2)
            elif op == "silu":
                value = (F.silu(src[0]*(2.0**exps[0]))/(2.0**eo)).round().clamp(*_bounds(bits))
            elif op == "relu":
                value = src[0].clamp_min(0)
            elif op == "concat":
                value = torch.cat([_float_requant(x, exp, eo,bits=bits) for x, exp in zip(src, exps)], dim=1)
            elif op == "requantize":
                value = _float_requant(src[0],exps[0],eo,bits=bits)
            else:
                raise ValueError("unsupported fake op: " + op)
            if list(value.shape[1:]) != node["output_shape"]:
                raise ValueError("fake node shape mismatch")
            values[name] = value
        return values

    def forward(self, images):
        values = self.forward_all(images)
        exps = {node["id"]:node["output_exp"] for node in self.graph["nodes"]}
        return dict(raw_predictions=tuple((values[self.graph["outputs"][level]] * (2.0**exps[self.graph["outputs"][level]])).float()
                                          for level in LEVELS))


@dataclass
class PreparedPTQ:
    folded_model: nn.Module
    fake_model: FakeReference
    integer_reference: IntegerReference
    extrema: dict
    profile: dict
    reference_outputs: dict = field(default_factory=dict)


@torch.no_grad()
def prepare_ptq(model, calibration, *, identity=None, checkpoint_binding=None, profile="W8A8", device="cpu"):
    if profile not in SUPPORTED_PROFILES:
        raise NotImplementedError("unsupported 192v2 precision profile: " + str(profile))
    contract, contract_hash = numeric_contract(profile)
    if model.training or not isinstance(calibration, TrainCalibration):
        raise ValueError("PTQ requires eval model and source-bound TrainCalibration")
    calibration.verify_manifest()
    if identity is not None:
        validate_identity(identity)
        if identity["profile"] != profile:
            raise ValueError("identity precision differs")
        if identity["mapping"]["numeric_contract_sha256"] != contract_hash:
            raise ValueError("mapping numeric contract differs from this PTQ implementation")
        if checkpoint_binding is not None and checkpoint_binding != identity["checkpoint"]:
            raise ValueError("duplicate checkpoint bindings differ")
        checkpoint_binding = deepcopy(identity["checkpoint"])
        if identity["calibration"].get("calibration_sha256", identity["calibration"].get("sha256")) != calibration.manifest["calibration_sha256"]:
            raise ValueError("identity calibration differs")
    if not isinstance(checkpoint_binding, dict) or not checkpoint_binding.get("sha256"):
        raise ValueError("actual source checkpoint binding required; initialized models cannot be training evidence")
    if not checkpoint_binding.get("path") or sha256_file(checkpoint_binding["path"]) != checkpoint_binding["sha256"]:
        raise ValueError("source checkpoint file missing or SHA256 differs")
    if checkpoint_binding.get("weights") != "ema" or checkpoint_binding.get("validation_weights") != "ema":
        raise ValueError("192v2 PTQ requires explicitly selected EMA weights")
    training_manifest = calibration.manifest.get("training_execution_manifest_path")
    if (not training_manifest or sha256_file(training_manifest) != calibration.manifest.get("training_execution_manifest_sha256")
            or calibration.manifest.get("training_execution_manifest_sha256") != checkpoint_binding.get("binding", {}).get("execution_manifest_sha256")):
        raise ValueError("calibration is not source-bound to the checkpoint training execution manifest")
    source_hash = model_state_hash(model)
    if checkpoint_binding.get("model_state_sha256") != source_hash:
        raise ValueError("checkpoint model-state binding differs")
    graph = build_graph(model)
    if identity is not None and any(identity[key] != graph[key] for key in ("space_version", "architecture_sha256", "full_contract_sha256")):
        raise ValueError("identity graph contract differs")
    if any(value.dtype != torch.float32 for value in model.state_dict().values() if value.is_floating_point()):
        raise ValueError("PTQ source must remain original FP32")
    folded = deepcopy(model).to(device=device, dtype=torch.float64).eval()
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(module, nn.BatchNorm2d) for module in folded.modules()):
        raise ValueError("unfolded batch normalization remains")
    statistics = {name: dict(output_max_abs=0.0) for name in ("image", *(node["id"] for node in graph["nodes"]))}
    first_witness, seen = None, 0
    for images in calibration.batches(device):
        values = float_graph(folded, graph, images.double())
        if first_witness is None:
            original = deepcopy(model).to(device=device, dtype=torch.float64).eval()
            native_before = raw_predictions(original, images.double())
            native_after = raw_predictions(folded, images.double())
            graph_after = [values[graph["outputs"][level]] for level in LEVELS]
            fold_errors = [float((a-b).abs().max()) for a,b in zip(native_before, native_after)]
            graph_errors = [float((a-b).abs().max()) for a,b in zip(native_after, graph_after)]
            if not all(torch.allclose(a,b,atol=1e-8,rtol=1e-8) for a,b in zip(native_before,native_after)):
                raise ValueError("BN fold mathematical FP64 parity failed")
            if any(value != 0 for value in graph_errors):
                raise ValueError("explicit graph differs from selected native raw model")
            first_witness = dict(input_images_sha256=tensor_sha(images), batch_images=len(images),
                                 bn_fold_max_errors=fold_errors, explicit_graph_max_errors=graph_errors,
                                 bn_fold_atol=1e-8,bn_fold_rtol=1e-8,graph_exact=True,
                                 scope="first_train_calibration_batch_FP64_mathematical_reference")
            del original, native_before, native_after, graph_after
        for name, value in values.items():
            peak = float(value.abs().max())
            if not math.isfinite(peak):
                raise ValueError("nonfinite calibration observation")
            statistics[name]["output_max_abs"] = max(statistics[name]["output_max_abs"], peak)
        seen += len(images)
        del values
    if seen != calibration.manifest["number_of_calibration_images"] or first_witness is None:
        raise ValueError("calibration count mismatch")
    if model_state_hash(model) != source_hash:
        raise ValueError("source model changed during PTQ")
    exps = {"image": exponent_from_maxabs(statistics["image"]["output_max_abs"])}
    numeric_graph, parameters, luts = deepcopy(graph), {}, {}
    head_bits, head_weight_bits = (8,8) if profile == "W8A8" else (16,16 if profile == "HEAD_W16A16" else 8)
    if profile != "W8A8":
        boundaries, new_nodes = {}, []
        for node in numeric_graph["nodes"]:
            if node["id"].startswith("head.cv2.") and node["id"].endswith(".0.conv"):
                index = int(node["id"].split(".")[2])
                name = "head.input_requant." + LEVELS[index]
                source = node["inputs"][0]
                boundary = dict(id=name,op="requantize",inputs=[source],input_shapes=deepcopy(node["input_shapes"]),
                                output_shape=list(node["input_shapes"][0]),activation="linear",
                                conversion="body_INT8_to_head_INT16_shared_by_regression_and_classification")
                boundaries[index] = name
                new_nodes.append(boundary)
                numeric_graph["edges"][name] = list(boundary["output_shape"])
                statistics[name] = deepcopy(statistics[source])
            if node["id"].startswith(("head.cv2.","head.cv3.")) and node["id"].endswith(".0.conv"):
                index = int(node["id"].split(".")[2])
                node["inputs"] = [boundaries[index]]
            new_nodes.append(node)
        if len(boundaries) != 4:
            raise ValueError("mixed head requires exactly four shared input conversion boundaries")
        numeric_graph["nodes"] = new_nodes
    numeric_graph.update(input_exp=exps["image"], numeric_contract_hash=contract_hash, precision_profile=profile,
                         precision_regions=dict(body=dict(input_bits=8,weight_bits=8,output_bits=8,
                                                          numeric_contract_hash=NUMERIC_CONTRACT_HASH),
                                                head=dict(input_bits=head_bits,weight_bits=head_weight_bits,output_bits=head_bits),
                                                boundary_count=0 if profile=="W8A8" else 4))
    if identity is not None:
        numeric_graph.update(identity_sha256=identity["identity_sha256"],
                             interface_contract_hash=identity["mapping"]["interface_contract_sha256"],
                             mapping=deepcopy(identity["mapping"]))
    for node in numeric_graph["nodes"]:
        name, op = node["id"], node["op"]
        inputs = [exps[source] for source in node["inputs"]]
        bits = head_bits if name.startswith("head.") else 8
        wb = head_weight_bits if name.startswith("head.") else 8
        output = inputs[0] if op in ("blur", "nearest", "maxpool", "relu") else (
            max(inputs) if op == "concat" else exponent_from_maxabs(statistics[name]["output_max_abs"],qmax=_bounds(bits)[1]))
        input_bits = 8 if op == "requantize" else bits
        node.update(input_exps=inputs, input_exp=inputs[0], output_exp=output, input_bits=input_bits, output_bits=bits,
                    numeric_contract_hash=contract_hash)
        if identity is not None:
            node["interface_contract_hash"] = identity["mapping"]["interface_contract_sha256"]
        exps[name] = output
        if op == "conv":
            conv = folded.get_submodule(name)
            we = exponent_from_maxabs(float(conv.weight.abs().max()),qmax=_bounds(wb)[1])
            weight = _quantize(conv.weight.detach().cpu().numpy(), we,bits=wb)
            bias = _quantize(conv.bias.detach().cpu().numpy(), inputs[0]+we, bias=True)
            m, s = h2.ratio(inputs[0]+we, output)
            node.update(weight_exp=we, weight_bits=wb, accumulator_bits=32, multiplier=m, shift=s, bias=True)
            parameters[name] = dict(weight=weight, bias=bias)
        elif op in ("add", "residual_add"):
            common = max(inputs)
            for value in inputs:
                h2.ratio(value, common)
            h2.ratio(common, output)
            node["common_exp"] = common
        elif op == "concat":
            for value in inputs:
                h2.ratio(value, output)
        elif op == "requantize":
            multiplier,shift = h2.ratio(inputs[0],output)
            node.update(multiplier=multiplier,shift=shift)
        elif op == "silu":
            table = silu_lut(inputs[0], output,input_bits=bits,output_bits=bits)
            lo,hi = _bounds(bits)
            independent = (F.silu(torch.arange(lo,hi+1,dtype=torch.float64)*(2.0**inputs[0]))/(2.0**output)).round().clamp(lo,hi).numpy().astype(_dtype(bits))
            if not np.array_equal(table, independent):
                raise ValueError("SiLU table differs from independent formula on the full signed input domain")
            luts[name] = table
            node.update(lut_sha256=array_sha(table), lut_entries=1 << bits, lut_index_offset=1 << (bits-1),
                        approximation="complete_signed_input_domain_quantized_SiLU_LUT")
    numeric_graph["numeric_graph_sha256"] = digest(numeric_graph)
    integer = IntegerReference(numeric_graph, parameters, luts)
    fake = FakeReference(numeric_graph, parameters).to(device).eval()
    sources = [Path(__file__), Path(graph_module.__file__), Path(h2.__file__)]
    from tools.modnas_comparison import quantization as calibration_module
    from hwnas_fpga.deploy import ptq_eval
    sources += [Path(calibration_module.__file__), Path(ptq_eval.__file__)]
    record = dict(schema_version="sonar192_ptq_v1", space_version=graph["space_version"],
                  architecture=graph["architecture"], architecture_sha256=graph["architecture_sha256"],
                  architecture_id=graph["architecture_id"], structure_id=graph["structure_id"], num_classes=2,
                  full_contract_sha256=graph["full_contract_sha256"], head_contract_sha256=graph["head_contract_sha256"],
                  precision_profile=profile, supported_profiles=list(SUPPORTED_PROFILES),
                  unsupported_profiles=[],
                  checkpoint_binding=deepcopy(checkpoint_binding), identity=deepcopy(identity),
                  identity_sha256=None if identity is None else identity["identity_sha256"],
                  calibration=deepcopy(calibration.manifest), calibration_sha256=calibration.manifest["calibration_sha256"],
                  numeric_contract=contract, numeric_contract_hash=contract_hash,
                  structural_graph_sha256=graph["structural_graph_sha256"], numeric_graph_sha256=numeric_graph["numeric_graph_sha256"],
                  original_model_state_sha256=source_hash, folded_model_state_sha256=model_state_hash(folded),
                  integer_parameters_sha256=integer.parameter_hash(), fake_model_state_sha256=model_state_hash(fake),
                  bn_folding=dict(pairs=pairs, witness=first_witness, dtype="torch.float64"),
                  source_sha256={str(path.resolve()):sha256_file(path) for path in sources},
                  integer_execution_kind=integer.execution_kind, fake_execution_kind=fake.execution_kind,
                  software_parity="NOT_RUN", quantized_accuracy="NOT_EVALUATED", hardware_verified=False,
                  formal_eligible=False, test_accessed=False, test_evaluated=False)
    return PreparedPTQ(folded, fake, integer, statistics, record)


@torch.no_grad()
def verify_parity(prepared, images):
    if (model_state_hash(prepared.fake_model) != prepared.profile["fake_model_state_sha256"]
            or prepared.integer_reference.parameter_hash() != prepared.profile["integer_parameters_sha256"]):
        raise ValueError("PTQ state drift before parity")
    for graph in (prepared.fake_model.graph, prepared.integer_reference.graph):
        body = deepcopy(graph)
        if body.pop("numeric_graph_sha256", None) != digest(body):
            raise ValueError("numeric graph hash drift before parity")
    if prepared.fake_model.graph != prepared.integer_reference.graph:
        raise ValueError("fake/integer graph identities differ")
    device = next(prepared.fake_model.buffers()).device
    fake = prepared.fake_model.forward_all(images.to(device))
    integer = prepared.integer_reference
    actual = integer.forward(integer.quantize_input(images))
    if set(fake) != set(actual):
        raise ValueError("parity node set differs")
    bits_by_node = {"image":8,**{node["id"]:node.get("output_bits",8) for node in integer.graph["nodes"]}}
    rows = []
    for name, value in actual.items():
        expected = fake[name].cpu().numpy()
        if not np.isfinite(expected).all() or not np.array_equal(expected, np.rint(expected)):
            raise ValueError("fake reference lost exact integer-valued property")
        lo,hi = _bounds(bits_by_node[name])
        if np.any(expected<lo) or np.any(expected>hi):
            raise ValueError("fake values outside registered signed output range")
        expected = expected.astype(_dtype(bits_by_node[name]))
        rows.append(dict(node=name, shape=list(value.shape), dtype=value.dtype.str, bits=bits_by_node[name],element_count=value.size,
                         mismatch_count=int(np.count_nonzero(value != expected)),
                         integer_sha256=array_sha(value), fake_quantized_sha256=array_sha(expected)))
    total = sum(row["mismatch_count"] for row in rows)
    status = "PASS_SOFTWARE_INTEGER_FAKE_PARITY" if total == 0 else "FAIL_SOFTWARE_INTEGER_FAKE_PARITY"
    prepared.profile["software_parity"] = status
    prepared.reference_outputs = {name:value.copy() for name,value in actual.items()}
    return dict(schema_version="sonar192_software_parity_v1", status=status,
                scope="all_nodes_independent_numpy_integer_vs_torch_FP64_fake_formula_SiLU",
                input_images_sha256=tensor_sha(images), input_images=len(images),
                numeric_graph_sha256=integer.graph["numeric_graph_sha256"],
                integer_parameters_sha256=integer.parameter_hash(), calibration_sha256=prepared.profile["calibration_sha256"],
                original_model_state_sha256=prepared.profile["original_model_state_sha256"],
                identity_sha256=prepared.profile["identity_sha256"], mismatches=total, rows=rows,
                hardware_verified=False, formal_eligible=False, test_evaluated=False)


def save_artifacts(output, prepared, *, witness):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("quantization exports require a new empty directory")
    if (witness.get("numeric_graph_sha256") != prepared.integer_reference.graph["numeric_graph_sha256"]
            or witness.get("integer_parameters_sha256") != prepared.integer_reference.parameter_hash()
            or not prepared.reference_outputs):
        raise ValueError("export requires actual bound software parity references")
    rows = {row["node"]:row for row in witness.get("rows", [])}
    if len(rows) != len(witness.get("rows", [])) or set(rows) != set(prepared.reference_outputs):
        raise ValueError("export reference node coverage differs")
    for name, value in prepared.reference_outputs.items():
        if (array_sha(value) != rows[name]["integer_sha256"] or list(value.shape) != rows[name]["shape"]
                or value.dtype != _dtype(rows[name].get("bits",8)) or value.size != rows[name]["element_count"]):
            raise ValueError("export reference bytes differ from measured witness")
    folder.mkdir(parents=True, exist_ok=True)
    payloads = {"profile.json":prepared.profile, "numeric_graph.json":prepared.integer_reference.graph,
                "extrema.json":prepared.extrema, "software_parity_witness.json":witness}
    torch.save(prepared.integer_reference.parameter_package(), folder/"integer_parameters.pt")
    arrays, entries = {}, []
    for name, row in prepared.integer_reference.parameters.items():
        for kind, value in row.items():
            key = "parameter_" + str(len(arrays))
            arrays[key] = value
            entries.append(dict(node=name, kind=kind, key=key, dtype=value.dtype.str,
                                shape=list(value.shape), sha256=array_sha(value)))
    for name, value in prepared.integer_reference.luts.items():
        key = "parameter_" + str(len(arrays))
        arrays[key] = value
        entries.append(dict(node=name, kind="lut", key=key, dtype=value.dtype.str,
                            shape=list(value.shape), sha256=array_sha(value)))
    np.savez_compressed(folder/"integer_parameters.npz", **arrays)
    payloads["parameter_index.json"] = dict(schema_version="sonar192_parameter_index_v1", parameters=entries)
    arrays, references = {}, []
    for name, value in prepared.reference_outputs.items():
        key = "node_" + str(len(arrays))
        arrays[key] = value
        references.append(dict(node=name,key=key,dtype=value.dtype.str,shape=list(value.shape),sha256=array_sha(value)))
    np.savez_compressed(folder/"reference_outputs.npz", **arrays)
    payloads["reference_index.json"] = dict(schema_version="sonar192_reference_index_v1",nodes=references,
        input_images_sha256=witness["input_images_sha256"], numeric_graph_sha256=witness["numeric_graph_sha256"],
        integer_parameters_sha256=witness["integer_parameters_sha256"])
    for name, payload in payloads.items():
        (folder/name).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    manifest = dict(schema_version="sonar192_quant_export_v1", files={path.name:sha256_file(path) for path in sorted(folder.iterdir()) if path.is_file()},
                    identity_sha256=prepared.profile["identity_sha256"], precision_profile=prepared.profile["precision_profile"],
                    numeric_graph_sha256=witness["numeric_graph_sha256"], software_parity=witness["status"],
                    hardware_verified=False, formal_eligible=False, test_evaluated=False)
    (folder/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    return folder
