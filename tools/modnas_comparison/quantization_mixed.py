"""Independent head-mixed software contracts for the NEW ELASTIC detector.

Uniform W8A8 delegates unchanged to the verified FP64-fold adapter. The body
always remains H2 NUMERIC_CONTRACT_V1. The four direct heads use the existing
H4 head_mixed_v1 arithmetic with explicit INT8-to-INT16 input conversions.
Those conversion nodes require new hardware evidence; software parity cannot
certify FPGA resources, HLS, frame/loader compatibility, or detection accuracy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import (
    H2FakeQuantConv, aligned_add_dequant, exponent_from_maxabs, power_of_two,
    quantize_weight, ratio_from_exponents, signed_range,
)
from tools.fpga_h2 import numeric as body_numeric
from tools.fpga_h4.quant_contracts import PROFILES, operator_contract, quant_contract
from tools.modnas_elastic.cost_graph import canonical_hash
from tools.modnas_elastic.model import LEVELS
from tools.modnas_comparison import quantization as body_adapter


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_VERSION = "elastic72_h2_body8_head_mixed_integer_v1"
SUPPORTED_PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")
TrainCalibration = body_adapter.TrainCalibration
model_state_hash = body_adapter.model_state_hash
PreparedElasticPTQ = body_adapter.PreparedElasticPTQ


def _dtype(bits):
    if bits not in (8, 16):
        raise ValueError("only signed INT8 and INT16 are registered")
    return np.dtype("i1" if bits == 8 else "<i2")


def _quantize_real(values, exponent, bits):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite mixed reference input")
    return np.clip(np.rint(values / power_of_two(exponent)), *signed_range(bits)).astype(_dtype(bits))


def requantize_int32(accumulator, multiplier, shift, *, bits):
    """H4 head range extension; never change V1's fixed INT8 function."""
    if bits == 8:
        return body_numeric.requantize_int32_to_int8(accumulator, multiplier, shift)
    if bits != 16:
        raise ValueError("unsupported mixed output bitwidth")
    values = np.asarray(accumulator, dtype=np.int64)
    if not 1 <= int(multiplier) <= 2**31 - 1:
        raise ValueError("mixed multiplier outside parent contract")
    if np.any(values < -2**31) or np.any(values > 2**31 - 1):
        raise ValueError("mixed requant input must already be wrapped INT32")
    # Signed INT32 times a valid multiplier is strictly within signed INT64.
    shifted = body_numeric.round_shift(values * int(multiplier), int(shift))
    return np.clip(shifted, -32768, 32767).astype("<i2")


def _contracts(profile):
    load_numeric_contract_v1()
    interface_path = ROOT / "reports/fpga_h2_contract/interface_contract_v1.json"
    closure_path = ROOT / "reports/fpga_h2_contract/closure_manifest.json"
    interface = json.loads(interface_path.read_text(encoding="utf-8"))
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    body = dict(numeric_contract_hash=EXPECTED_HASH, interface_contract_hash=canonical_hash(interface))
    if (closure["numeric_contract_hash"] != body["numeric_contract_hash"]
            or closure["interface_contract_hash"] != body["interface_contract_hash"]):
        raise ValueError("body interface/numeric declaration changed from frozen H2 closure")
    wb, ab = PROFILES[profile]
    head_payload = dict(parent=EXPECTED_HASH, version="head_mixed_v1",
        input_bits=ab, output_bits=ab, weight_bits=wb, accumulator_bits=32,
        overflow="twos_complement_modulo_2pow32", rounding="nearest_even", zero_point=0,
        requant_product_bits=64, scale="positive_power_of_two_exponent_int32")
    head = operator_contract(wb, ab, body)
    if canonical_hash(head_payload) != head["numeric_contract_hash"]:
        raise ValueError("mixed arithmetic differs from the existing H4 head contract")
    boundary = dict(version="elastic_body8_to_head16_requant_v1",
        parent_body_numeric_contract_hash=EXPECTED_HASH,
        destination_head_numeric_contract_hash=head["numeric_contract_hash"],
        input_bits=8, output_bits=16, arithmetic_bits=64, input_accumulator_bits=32,
        rounding="nearest_even", saturation=[-32768, 32767], zero_point=0,
        scale="positive_power_of_two_exponent_int32", source="actual_INT8_body_edge",
        conversion="integer_requantize_source_scale_to_head_input_scale")
    return dict(body=body, head=head, head_numeric_payload=head_payload,
                boundary_numeric_payload=boundary, boundary_numeric_contract_hash=canonical_hash(boundary),
                candidate=quant_contract(profile, body))


class ExactValuedMixedHeadConv(H2FakeQuantConv):
    """Exact-valued FP64 head simulation with INT32 wrap; not integer runtime."""
    def forward(self, inputs):
        if (self.input_bits, self.weight_bits, self.output_bits) not in ((16, 8, 16), (16, 16, 16)):
            raise ValueError("registered mixed head must be I16/W8or16/O16")
        if self.weight_granularity != "per_tensor":
            raise ValueError("registered mixed head requires per-tensor weights")
        values = inputs.double().div(power_of_two(self.input_exp)).round().clamp(*signed_range(16))
        products = int(np.prod(self.int_weight.shape[1:]))
        bound = products * (1 << 15) * (1 << (self.weight_bits - 1)) + 2**31
        if bound >= 2**53:
            raise ValueError("FP64 fake head lacks an exact integer-valued accumulation bound")
        accumulator = F.conv2d(values, self.int_weight.double(), self.int_bias.double(),
                               stride=self.stride, padding=self.padding,
                               dilation=self.dilation, groups=self.groups)
        accumulator = torch.remainder(accumulator + 2**31, 2**32) - 2**31
        quantized = (accumulator * self.multiplier / (1 << self.shift)).round().clamp(-32768, 32767)
        if self.activation != "linear":
            raise ValueError("new direct detection heads must preserve linear logits")
        return (quantized * power_of_two(self.output_exp)).float()


class MixedFakeQuantElasticGraph(nn.Module):
    def __init__(self, folded, graph):
        super().__init__()
        self.graph = deepcopy(graph)
        self.paths, self.convs = {}, nn.ModuleDict()
        for node in self.graph["nodes"]:
            if node["op"] != "conv":
                continue
            key = "conv" + str(len(self.convs))
            self.paths[node["id"]] = key
            cls = ExactValuedMixedHeadConv if node["id"].startswith("head.") else body_adapter.ExactValuedH2FakeQuantConv
            self.convs[key] = cls(folded.get_submodule(node["id"]),
                input_exp=node["input_exp"], output_exp=node["output_exp"], weight_exps=[node["weight_exp"]],
                activation=node["activation"], weight_granularity="per_tensor",
                input_bits=node["input_bits"], weight_bits=node["weight_bits"], output_bits=node["output_bits"])
        self.execution_kind = "FLOAT64_EXACT_VALUED_BODY8_HEAD16_INT32_WRAP_NOT_INTEGER_RUNTIME"

    def forward_all(self, images):
        body_adapter._check_images(images)
        exponent = self.graph["input_exp"]
        values = {"image": images.double().div(power_of_two(exponent)).round().clamp(-128, 127)
                  .mul(power_of_two(exponent)).float()}
        for node in self.graph["nodes"]:
            sources = [values[path] for path in node["inputs"]]
            if node["op"] == "conv":
                value = self.convs[self.paths[node["id"]]](sources[0])
            elif node["op"] == "requantize":
                integers = sources[0].double().div(power_of_two(node["input_exp"])).round()
                value = (integers * node["multiplier"] / (1 << node["shift"])).round().clamp(-32768, 32767)
                value = (value * power_of_two(node["output_exp"])).float()
            elif node["op"] in {"add", "residual_add"}:
                value = aligned_add_dequant(sources[0], sources[1], node["input_exp_a"],
                                           node["input_exp_b"], node["output_exp"])
            elif node["op"] == "nearest":
                value = F.interpolate(sources[0], size=tuple(node["output_shape"][-2:]), mode="nearest")
            elif node["op"] == "maxpool":
                value = F.max_pool2d(sources[0], node["kernel"], node["stride"], node["padding"])
            else:
                raise ValueError("unsupported mixed primitive: " + node["op"])
            values[node["id"]] = value
        return values

    def forward(self, images):
        values = self.forward_all(images)
        return dict(predictions=tuple(values[self.graph["outputs"][level]] for level in LEVELS))


@dataclass
class MixedIntegerElasticResult:
    outputs: dict
    output_exponents: dict
    values: dict
    trace: list

    @property
    def predictions(self):
        return dict(predictions=tuple(torch.from_numpy(self.outputs[level].copy()).float()
                                     * power_of_two(self.output_exponents[level]) for level in LEVELS))


class MixedIntegerElasticReference(body_adapter.IntegerElasticReference):
    def __init__(self, fake):
        self.graph = deepcopy(fake.graph)
        self.parameters = {}
        for path, key in fake.paths.items():
            conv = fake.convs[key]
            self.parameters[path] = dict(weight=conv.int_weight.detach().cpu().numpy().astype(_dtype(conv.weight_bits), copy=True),
                                         bias=conv.int_bias.detach().cpu().numpy().astype("<i4", copy=True))
        self.execution_kind = "NUMPY_INTEGER_BODY_INT8_HEAD_INT16_INT64_SUM_INT32_WRAP_INT64_REQUANT"

    def forward(self, images_int8, *, retain=False):
        if isinstance(images_int8, torch.Tensor):
            images_int8 = images_int8.detach().cpu().numpy()
        images = np.asarray(images_int8)
        if images.dtype != np.int8 or images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("mixed runtime begins with already quantized signed INT8 [B,1,640,640]")
        batched = {}
        for image in images:
            values = {"image": image}
            for node in self.graph["nodes"]:
                sources = [values[path] for path in node["inputs"]]
                if node["op"] == "conv":
                    params = self.parameters[node["id"]]
                    if sources[0].dtype != _dtype(node["input_bits"]) or params["weight"].dtype != _dtype(node["weight_bits"]):
                        raise ValueError("mixed convolution input/weight dtype differs at " + node["id"])
                    if node["output_bits"] == 8:
                        value = body_numeric.conv(sources[0], params["weight"], params["bias"], node)
                    else:
                        accumulator = body_numeric.conv_acc(sources[0], params["weight"], params["bias"],
                            stride=node["stride"], padding=node["padding"], groups=node["groups"])
                        value = requantize_int32(accumulator, node["multiplier"], node["shift"], bits=16)
                elif node["op"] == "requantize":
                    if sources[0].dtype != np.int8:
                        raise ValueError("head input conversion must read an actual INT8 body edge")
                    value = requantize_int32(sources[0].astype(np.int32), node["multiplier"], node["shift"], bits=16)
                elif node["op"] in {"add", "residual_add"}:
                    value = body_numeric.aligned_add(sources[0], sources[1], node["input_exp_a"],
                                                   node["input_exp_b"], node["output_exp"])
                elif node["op"] == "nearest":
                    value = np.repeat(np.repeat(sources[0], 2, axis=1), 2, axis=2)
                elif node["op"] == "maxpool":
                    channels, height, width = sources[0].shape
                    value = sources[0].reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))
                else:
                    raise ValueError("unsupported mixed integer primitive")
                if value.dtype != _dtype(node["output_bits"]) or tuple(value.shape) != tuple(node["output_shape"]):
                    raise ValueError("mixed integer dtype/geometry differs at " + node["id"])
                values[node["id"]] = value
            saved = values if retain else {path: values[path] for path in self.graph["outputs"].values()}
            for path, value in saved.items():
                batched.setdefault(path, []).append(value)
        stacked = {path: np.stack(values) for path, values in batched.items()}
        nodes = {node["id"]: node for node in self.graph["nodes"]}
        trace = [dict(node=path, op="image" if path == "image" else nodes[path]["op"], dtype=str(value.dtype),
                      element_count=int(value.size), integer_sha256=hashlib.sha256(value.tobytes()).hexdigest())
                 for path, value in stacked.items()]
        return MixedIntegerElasticResult(
            outputs={level: stacked[path] for level, path in self.graph["outputs"].items()},
            output_exponents={level: nodes[path]["output_exp"] for level, path in self.graph["outputs"].items()},
            values=stacked if retain else {}, trace=trace)

    def parameter_package(self):
        return dict(numeric_contract="H2_BODY_V1_WITH_SEPARATE_H4_HEAD_MIXED_V1_AND_BOUNDARY_V1",
                    numeric_contract_hash=self.graph["candidate_numeric_contract_hash"],
                    numeric_contract_regions=deepcopy(self.graph["numeric_contract_regions"]),
                    graph=deepcopy(self.graph),
                    weights={path: torch.from_numpy(record["weight"].copy()) for path, record in self.parameters.items()},
                    biases={path: torch.from_numpy(record["bias"].copy()) for path, record in self.parameters.items()},
                    execution_kind=self.execution_kind,
                    hls_parity="NOT_RUN", board_parity="NOT_RUN", hardware_feasibility="UNKNOWN")


def _prepare_from_body(model, calibration, body, profile, device):
    contract = _contracts(profile)
    if (model_state_hash(model) != body.profile["original_model_state_sha256"]
            or calibration.manifest != body.profile["calibration"]
            or model_state_hash(body.folded_model) != body.profile["folded_model_state_sha256"]):
        raise ValueError("source model/calibration/folded reference changed before mixed preparation")
    graph = deepcopy(body.integer_reference.graph)
    del graph["numeric_graph_sha256"]
    graph.update(precision_profile=profile, numeric_contract_hash=EXPECTED_HASH,
                 candidate_numeric_contract_hash=contract["candidate"]["quant_contract_hash"],
                 numeric_contract_regions=contract,
                 graph_scope="new_software_mixed_integer_topology_with_unmeasured_body8_to_head16_nodes",
                 quantization_status="MIXED_SOFTWARE_REFERENCE_PREPARED_PARITY_NOT_YET_RUN")
    nodes, new_layers = [], []
    head_w, head_a = PROFILES[profile]
    for original in graph["nodes"]:
        node = deepcopy(original)
        node.update(input_bits=8, weight_bits=8 if node["op"] == "conv" else 0, output_bits=8,
                    numeric_contract_hash=EXPECTED_HASH)
        if node["op"] == "conv" and node["id"].startswith("head."):
            conv = body.folded_model.get_submodule(node["id"])
            observation = body.extrema[node["id"]]
            source_exp = node["input_exp"]
            input_exp = exponent_from_maxabs(observation["input_max_abs"], qmax=32767)
            output_exp = exponent_from_maxabs(observation["output_max_abs"], qmax=32767)
            _, weight_exps = quantize_weight(conv.weight, granularity="per_tensor", bits=head_w)
            boundary_id = node["id"].rsplit(".", 1)[0] + ".input_requant"
            multiplier, shift = ratio_from_exponents(source_exp, input_exp)
            boundary = dict(id=boundary_id, op="requantize", operator_type="body8_to_head16_requantize",
                inputs=list(node["inputs"]), input_shapes=deepcopy(node["input_shapes"]),
                output_shape=list(node["input_shapes"][0]), input_exp=source_exp, output_exp=input_exp,
                input_bits=8, weight_bits=0, output_bits=16, multiplier=multiplier, shift=shift,
                numeric_contract_hash=contract["boundary_numeric_contract_hash"],
                activation="linear", hardware_measurement="MISSING", software_parameters=0)
            nodes.append(boundary)
            graph["edges"][boundary_id] = list(boundary["output_shape"])
            multiplier, shift = ratio_from_exponents(input_exp + weight_exps[0], output_exp)
            node.update(inputs=[boundary_id], input_exp=input_exp, weight_exp=weight_exps[0],
                output_exp=output_exp, multiplier=multiplier, shift=shift,
                input_bits=head_a, weight_bits=head_w, output_bits=head_a,
                numeric_contract_hash=contract["head"]["numeric_contract_hash"],
                interface_contract_hash=contract["head"]["interface_contract_hash"])
        nodes.append(node)
        if node["op"] == "conv":
            record = next(deepcopy(row) for row in body.profile["layers"] if row["name"] == node["id"])
            record.update({key: node[key] for key in ("input_exp", "weight_exp", "output_exp", "multiplier", "shift",
                                                     "input_bits", "weight_bits", "output_bits", "numeric_contract_hash")})
            record.update(input_saturation=list(signed_range(node["input_bits"])),
                          weight_saturation=list(signed_range(node["weight_bits"])),
                          output_saturation=list(signed_range(node["output_bits"])))
            new_layers.append(record)
    graph["nodes"] = nodes
    graph["numeric_graph_sha256"] = canonical_hash(graph)
    fake = MixedFakeQuantElasticGraph(body.folded_model, graph).to(device).eval()
    integer = MixedIntegerElasticReference(fake)
    record = deepcopy(body.profile)
    record.pop("signed_range")
    record.update(adapter_version=ADAPTER_VERSION, body_adapter_version=body.profile["adapter_version"],
        precision_profile=profile, supported_profiles=list(SUPPORTED_PROFILES), unsupported_profiles=[],
        numeric_contract="H2_BODY_V1_WITH_SEPARATE_H4_HEAD_MIXED_V1_AND_BOUNDARY_V1",
        numeric_contract_hash=contract["candidate"]["quant_contract_hash"], numeric_contract_regions=contract,
        bitwidth=dict(body_activation=8, body_weight=8, head_input=16, head_activation=16, head_weight=head_w,
                      accumulator=32, requant_product=64),
        saturation_by_region=dict(body=[-128,127], head=[-32768,32767]),
        layers=new_layers, numeric_graph_sha256=graph["numeric_graph_sha256"],
        head_input_conversions=[deepcopy(node) for node in nodes if node["op"] == "requantize"],
        output_exponents={level: next(node["output_exp"] for node in nodes if node["id"] == path)
                          for level, path in graph["outputs"].items()},
        fake_execution_kind=fake.execution_kind, integer_execution_kind=integer.execution_kind,
        fake_model_state_sha256=model_state_hash(fake), integer_parameters_sha256=integer.parameter_hash(),
        integer_vs_fake_parity="NOT_RUN", mixed_head_integer_arithmetic_implemented=True,
        hardware_boundary_nodes_measured=False,
        full_hardware_export="NOT_IMPLEMENTED_NEW_HEAD16_BOUNDARIES_FRAME_AND_LOADER")
    record["bn_folding"]["witness"]["observed_output_LSB_profile"] = "W8A8_body_reference_observation"
    sources = (Path(__file__), Path(body_adapter.__file__), ROOT / "tools/fpga_h4/quant_contracts.py",
               ROOT / "reports/fpga_h2_contract/interface_contract_v1.json",
               ROOT / "reports/fpga_h2_contract/closure_manifest.json")
    record["adapter_sources_sha256"].update({str(path.resolve()): body_adapter._file_sha(path) for path in sources})
    return PreparedElasticPTQ(body.folded_model, fake, integer, deepcopy(body.extrema), record)


@torch.no_grad()
def prepare_elastic_ptq(model, calibration, *, device="cpu", profile="W8A8"):
    if profile not in SUPPORTED_PROFILES:
        raise NotImplementedError("unsupported software precision profile: " + str(profile))
    body = body_adapter.prepare_elastic_ptq(model, calibration, device=device, profile="W8A8")
    return body if profile == "W8A8" else _prepare_from_body(model, calibration, body, profile, device)


@torch.no_grad()
def prepare_elastic_profiles(model, calibration, *, device="cpu"):
    """Observe train64 once; bind all profiles to that same untouched source."""
    body = body_adapter.prepare_elastic_ptq(model, calibration, device=device, profile="W8A8")
    return {"W8A8": body, **{profile: _prepare_from_body(model, calibration, body, profile, device)
                            for profile in SUPPORTED_PROFILES[1:]}}


@torch.no_grad()
def verify_fake_integer_parity(prepared, images):
    if prepared.profile["precision_profile"] == "W8A8":
        return body_adapter.verify_fake_integer_parity(prepared, images)
    fake, integer = prepared.fake_model, prepared.integer_reference
    if (model_state_hash(fake) != prepared.profile["fake_model_state_sha256"]
            or integer.parameter_hash() != prepared.profile["integer_parameters_sha256"]):
        raise ValueError("mixed PTQ parameter drift before parity measurement")
    for graph in (fake.graph, integer.graph):
        actual = deepcopy(graph)
        declared = actual.pop("numeric_graph_sha256")
        if declared != prepared.profile["numeric_graph_sha256"] or canonical_hash(actual) != declared:
            raise ValueError("mixed numeric graph drift before parity measurement")
    values = fake.forward_all(images.to(next(fake.buffers()).device))
    result = integer.forward(integer.quantize_input(images), retain=True)
    nodes = {node["id"]: node for node in integer.graph["nodes"]}
    rows = []
    for path, expected in result.values.items():
        bits = 8 if path == "image" else nodes[path]["output_bits"]
        exponent = integer.graph["input_exp"] if path == "image" else nodes[path]["output_exp"]
        actual = _quantize_real(values[path].detach().cpu().numpy(), exponent, bits)
        rows.append(dict(node=path, output_bits=bits, dtype=str(expected.dtype), element_count=int(expected.size),
            mismatch_count=int(np.count_nonzero(expected != actual)),
            integer_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
            fake_quantized_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    total = sum(row["mismatch_count"] for row in rows)
    return dict(status="PASS_SOFTWARE_INTEGER_FAKE_PARITY" if total == 0 else "FAIL_SOFTWARE_INTEGER_FAKE_PARITY",
        scope="independent_numpy_mixed_integer_vs_exact_valued_FP64_fake_on_true_new_DAG_with_head_input_conversions",
        precision_profile=prepared.profile["precision_profile"],
        numeric_contract_hash=prepared.profile["numeric_contract_hash"],
        numeric_contract_regions=deepcopy(prepared.profile["numeric_contract_regions"]),
        input_images_sha256=body_adapter._tensor_sha(images), input_images=int(images.shape[0]),
        calibration_sha256=prepared.profile["calibration"]["calibration_sha256"],
        original_model_state_sha256=prepared.profile["original_model_state_sha256"],
        numeric_graph_sha256=integer.graph["numeric_graph_sha256"],
        integer_parameters_sha256=prepared.profile["integer_parameters_sha256"], mismatches=total, rows=rows,
        formal_eligible=False, integer_vs_HLS_parity="NOT_RUN", board_parity="NOT_RUN", test_evaluated=False)


def save_ptq_artifacts(output, prepared, *, witness=None):
    return body_adapter.save_ptq_artifacts(output, prepared, witness=witness)


__all__ = ["ADAPTER_VERSION", "SUPPORTED_PROFILES", "TrainCalibration", "model_state_hash",
           "PreparedElasticPTQ", "prepare_elastic_ptq", "prepare_elastic_profiles", "verify_fake_integer_parity",
           "save_ptq_artifacts", "requantize_int32", "ExactValuedMixedHeadConv"]
