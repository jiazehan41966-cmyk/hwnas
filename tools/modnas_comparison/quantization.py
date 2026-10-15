"""Source-bound, train-only PTQ and integer reference for the new ELASTIC graph.

This adapter executes the real recurrent top-down and bottom-up topology. It
never replaces the neck with the historical FixedFPN. H2FakeQuantConv remains
a floating simulation, while IntegerElasticReference uses the retained H2
NumPy integer arithmetic (INT64 sums, INT32 modulo, INT64 requantization).
Neither preparing a profile nor software parity establishes HLS or board
parity, quantized accuracy retention, or hardware feasibility.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
from hwnas_fpga.deploy.h2_ptq import (
    H2FakeQuantConv, aligned_add_dequant, exponent_from_maxabs,
    power_of_two, quantize_weight, ratio_from_exponents,
)
from hwnas_fpga.deploy.ptq_eval import fold_batch_norms_inplace
from tools.fpga_h2 import numeric as integer_numeric
from tools.modnas_elastic.cost_graph import build_cost_graph, canonical_hash
from tools.modnas_elastic.model import LEVELS, StaticElasticNeckDetector


ADAPTER_VERSION = "elastic72_train_bound_ptq_w8a8_fp64_bn_reference_v2"
SUPPORTED_PROFILES = ("W8A8",)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)],
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(bytes.fromhex(_tensor_sha(value)))
    return digest.hexdigest()


class TrainCalibration:
    """Decode only a SHA-verified, frozen train prefix; default is exactly 64.

    ``formal=False`` permits a smaller source-bound integration fixture. It
    does not turn fixture calibration into formal experiment evidence.
    """
    def __init__(self, dataset_root, frozen_train_entries, *, batch_size=1,
                 num_images=64, formal=True):
        self.root = Path(dataset_root).resolve()
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if type(num_images) is not int or num_images < 1:
            raise ValueError("num_images must be a positive integer")
        if type(formal) is not bool or (formal and num_images != 64):
            raise ValueError("formal PTQ requires exactly the frozen first 64 train images")
        rows = list(deepcopy(frozen_train_entries))
        if len(rows) < num_images:
            raise ValueError("frozen train has fewer images than the requested calibration prefix")
        # Reject split metadata before ordering checks or opening any source.
        for row in rows:
            self._path(row.get("image"), "images")
            self._path(row.get("annotation"), "labels")
        names = [row.get("image") for row in rows]
        if any(not isinstance(name, str) for name in names) or names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("frozen train entries must be unique in sorted loader order")
        self.batch_size, self.formal = batch_size, formal
        self.entries = rows[:num_images]
        for entry in self.entries:
            self._verify_entry(entry)
        self.manifest = dict(
            split="train", augmentation=False, shuffle=False,
            valid_used_for_selection=False, test_accessed=False, test_evaluated=False,
            input_chw=[1, 640, 640], decoding="PIL_L_BILINEAR_resize_divide255_no_augmentation",
            order="frozen_loader_sorted_first_64" if formal else "integration_fixture_frozen_train_prefix",
            number_of_calibration_images=num_images, batch_size=batch_size,
            dataset_root=str(self.root), entries=deepcopy(self.entries),
            frozen_train_entries_sha256=canonical_hash(rows),
            formal_calibration_count=formal, formal_eligible=False,
        )
        self.manifest["calibration_sha256"] = canonical_hash(self.manifest)

    def _path(self, name, kind):
        if not isinstance(name, str):
            raise ValueError("calibration source path must be a train-relative string")
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (path.is_absolute() or ".." in path.parts or len(path.parts) < 3
                or tuple(path.parts[:2]) != ("train", kind)
                or any(part.lower() in {"valid", "validation", "val", "test"} for part in path.parts)):
            raise ValueError("calibration is train-only; valid/test paths are forbidden")
        resolved = (self.root / Path(*path.parts)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("calibration source resolves outside the frozen dataset root")
        return resolved

    def _verify_entry(self, entry):
        for name, kind, sha_key in (("image", "images", "image_sha256"),
                                    ("annotation", "labels", "annotation_sha256")):
            path = self._path(entry.get(name), kind)
            expected = entry.get(sha_key)
            if not isinstance(expected, str) or len(expected) != 64 or _file_sha(path) != expected:
                raise ValueError("calibration source SHA256 drift: " + name)

    def batches(self, device="cpu"):
        self.verify_manifest()
        batch = []
        for entry in self.entries:
            self._verify_entry(entry)
            with Image.open(self._path(entry["image"], "images")) as image:
                image = image.convert("L").resize((640, 640), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
            batch.append(torch.from_numpy(array.copy()).unsqueeze(0))
            if len(batch) == self.batch_size:
                yield torch.stack(batch).to(device)
                batch = []
        if batch:
            yield torch.stack(batch).to(device)

    def verify_manifest(self):
        body = deepcopy(self.manifest)
        bound_hash = body.pop("calibration_sha256", None)
        if (canonical_hash(body) != bound_hash or self.entries != body.get("entries")
                or body.get("split") != "train" or body.get("dataset_root") != str(self.root)
                or any(body.get(key) is not False for key in
                       ("augmentation", "shuffle", "valid_used_for_selection", "test_accessed", "test_evaluated"))):
            raise ValueError("source-bound train calibration manifest drift")


def _activation(tensor, kind):
    if kind == "linear":
        return tensor
    if kind == "relu":
        return tensor.clamp_min(0)
    if kind == "relu6":
        return tensor.clamp(0, 6)
    raise ValueError("unsupported graph activation: " + kind)


def _check_images(images):
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
        raise ValueError("ELASTIC integer/PTQ graph requires [B,1,640,640]")


def _float_graph(model, graph, images):
    """Execute explicit true graph, retaining outputs of every primitive."""
    _check_images(images)
    values = {"image": images}
    for node in graph["nodes"]:
        sources = [values[name] for name in node["inputs"]]
        if node["op"] == "conv":
            value = _activation(model.get_submodule(node["id"])(sources[0]), node["activation"])
        elif node["op"] in {"add", "residual_add"}:
            value = sources[0] + sources[1]
        elif node["op"] == "nearest":
            value = F.interpolate(sources[0], size=tuple(node["output_shape"][-2:]), mode="nearest")
        elif node["op"] == "maxpool":
            value = F.max_pool2d(sources[0], node["kernel"], node["stride"], node["padding"])
        else:
            raise ValueError("unsupported new graph primitive: " + node["op"])
        if tuple(value.shape[1:]) != tuple(node["output_shape"]):
            raise ValueError("actual graph geometry differs at " + node["id"])
        values[node["id"]] = value
    return values


def _prediction_errors(reference, actual):
    if len(reference) != 4 or len(actual) != 4:
        raise ValueError("ELASTIC output must preserve four prediction heads")
    return [float((left - right).abs().max()) for left, right in zip(reference, actual)]


def _prediction_comparison(reference, actual, *, atol, rtol):
    """Keep measured rounding drift separate from a mathematical fold gate."""
    rows = []
    for level, left, right in zip(LEVELS, reference, actual):
        difference = (left.double() - right.double()).abs()
        peak = float(left.abs().max())
        maximum = float(difference.max())
        rows.append(dict(level=level, max_abs_error=maximum,
                         reference_max_abs=peak,
                         peak_normalized_max_error=maximum / max(peak, 1e-300),
                         elements_outside_tolerance=int(
                             (difference > atol + rtol * left.double().abs()).sum()),
                         allclose=bool(torch.allclose(left.double(), right.double(), atol=atol, rtol=rtol))))
    return dict(atol=atol, rtol=rtol, all_heads_allclose=all(row["allclose"] for row in rows), heads=rows)


def _validate_folded_graph(model, graph):
    actual = {name: module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
    expected = {node["id"]: node for node in graph["nodes"] if node["op"] == "conv"}
    if set(actual) != set(expected):
        raise ValueError("new graph convolution membership differs from selected static model")
    for name, conv in actual.items():
        node = expected[name]
        if (conv.in_channels != node["Cin"] or conv.out_channels != node["Cout"]
                or conv.kernel_size != (node["kernel"],) * 2
                or conv.stride != (node["stride"],) * 2
                or conv.padding != (node["padding"],) * 2
                or conv.groups != node["groups"] or conv.dilation != (1, 1)):
            raise ValueError("new graph convolution geometry differs at " + name)
        if conv.bias is None:
            raise ValueError("folded convolution lacks explicit bias at " + name)


class ExactValuedH2FakeQuantConv(H2FakeQuantConv):
    """Reuse H2 quantized parameters with exact-valued FP64 simulation.

    Historical H2FakeQuantConv uses FP32 accumulation and does not implement
    INT32 overflow. Here INT8 products and sums plus an INT32 bias are exact
    in FP64 for this small graph; modulo and nearest-even are explicit. This
    remains floating arithmetic, independently checked against NumPy integers.
    """
    def forward(self, inputs):
        if (self.input_bits, self.weight_bits, self.output_bits) != (8, 8, 8):
            raise NotImplementedError("exact-valued new graph currently admits W8A8 only")
        if self.weight_granularity != "per_tensor":
            raise NotImplementedError("exact-valued new graph requires per-tensor H2 weights")
        values = inputs.double().div(power_of_two(self.input_exp)).round().clamp(-128, 127)
        weight, bias = self.int_weight.double(), self.int_bias.double()
        # A conservative bound proves every mathematical sum is below FP64's
        # exact integer range, regardless of the actual input/weight values.
        products = int(np.prod(self.int_weight.shape[1:]))
        if products * 128 * 128 + 2**31 >= 2**53:
            raise ValueError("FP64 fake accumulation lacks an exact-valued integer bound")
        accumulator = F.conv2d(values, weight, bias, stride=self.stride,
                               padding=self.padding, dilation=self.dilation, groups=self.groups)
        accumulator = torch.remainder(accumulator + 2**31, 2**32) - 2**31
        requantized = (accumulator * self.multiplier / (1 << self.shift)).round().clamp(-128, 127)
        if self.activation == "relu":
            requantized = requantized.clamp_min(0)
        elif self.activation == "relu6":
            requantized = requantized.clamp(0, integer_numeric.q6(self.output_exp))
        elif self.activation != "linear":
            raise ValueError("unsupported H2 activation")
        return (requantized * power_of_two(self.output_exp)).float()


class FakeQuantElasticGraph(nn.Module):
    """Exact-valued FP64 H2 simulation, explicitly not integer runtime."""
    def __init__(self, folded, numeric_graph):
        super().__init__()
        self.graph = deepcopy(numeric_graph)
        self.convs = nn.ModuleDict()
        self.paths = {}
        for node in self.graph["nodes"]:
            if node["op"] != "conv":
                continue
            key = "conv" + str(len(self.convs))
            conv = folded.get_submodule(node["id"])
            self.paths[node["id"]] = key
            self.convs[key] = ExactValuedH2FakeQuantConv(
                conv, input_exp=node["input_exp"], output_exp=node["output_exp"],
                weight_exps=[node["weight_exp"]], activation=node["activation"],
                weight_granularity="per_tensor")
        self.execution_kind = "FLOAT64_EXACT_VALUED_H2_FAKE_INT32_WRAP_NOT_INTEGER_RUNTIME"

    def forward_all(self, images):
        _check_images(images)
        exponent = self.graph["input_exp"]
        # Input quantization is part of the graph and has a single bound exponent.
        images = images.div(power_of_two(exponent)).round().clamp(-128, 127) * power_of_two(exponent)
        values = {"image": images}
        for node in self.graph["nodes"]:
            sources = [values[name] for name in node["inputs"]]
            if node["op"] == "conv":
                value = self.convs[self.paths[node["id"]]](sources[0])
            elif node["op"] in {"add", "residual_add"}:
                value = aligned_add_dequant(sources[0], sources[1], node["input_exp_a"],
                                           node["input_exp_b"], node["output_exp"])
            elif node["op"] == "nearest":
                value = F.interpolate(sources[0], size=tuple(node["output_shape"][-2:]), mode="nearest")
            elif node["op"] == "maxpool":
                value = F.max_pool2d(sources[0], node["kernel"], node["stride"], node["padding"])
            else:
                raise ValueError("unsupported primitive")
            values[node["id"]] = value
        return values

    def forward(self, images):
        values = self.forward_all(images)
        return dict(predictions=tuple(values[self.graph["outputs"][level]] for level in LEVELS))


@dataclass
class IntegerElasticResult:
    outputs: dict[str, np.ndarray]
    output_exponents: dict[str, int]
    values: dict[str, np.ndarray]
    trace: list[dict[str, Any]]

    @property
    def predictions(self):
        return dict(predictions=tuple(torch.from_numpy(self.outputs[level].copy()).float()
                                     * power_of_two(self.output_exponents[level]) for level in LEVELS))


class IntegerElasticReference:
    """True signed INT8 graph using source-bound H2 arithmetic and parameters."""
    def __init__(self, fake_model: FakeQuantElasticGraph):
        self.graph = deepcopy(fake_model.graph)
        self.parameters = {}
        for path, key in fake_model.paths.items():
            module = fake_model.convs[key]
            self.parameters[path] = dict(weight=module.int_weight.detach().cpu().numpy().astype("i1", copy=True),
                                         bias=module.int_bias.detach().cpu().numpy().astype("<i4", copy=True))
        self.execution_kind = "NUMPY_INTEGER_ONLY_INT64_SUM_INT32_WRAP_INT64_REQUANT_INT8"

    def quantize_input(self, images):
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        images = np.asarray(images)
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("quantize_input requires [B,1,640,640]")
        return integer_numeric.quantize_real(images, self.graph["input_exp"])

    def forward(self, images_int8, *, retain=False):
        if isinstance(images_int8, torch.Tensor):
            images_int8 = images_int8.detach().cpu().numpy()
        images = np.asarray(images_int8)
        if images.dtype != np.int8 or images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("integer runtime requires already quantized signed INT8 [B,1,640,640]")
        batched_values, traces = {}, []
        for image in images:
            values = {"image": image}
            for node in self.graph["nodes"]:
                sources = [values[name] for name in node["inputs"]]
                if node["op"] == "conv":
                    parameters = self.parameters[node["id"]]
                    value = integer_numeric.conv(sources[0], parameters["weight"], parameters["bias"], node)
                elif node["op"] in {"add", "residual_add"}:
                    value = integer_numeric.aligned_add(sources[0], sources[1], node["input_exp_a"],
                                                       node["input_exp_b"], node["output_exp"])
                elif node["op"] == "nearest":
                    if tuple(node["output_shape"][-2:]) != (sources[0].shape[1] * 2, sources[0].shape[2] * 2):
                        raise ValueError("integer nearest requires exact factor two")
                    value = np.repeat(np.repeat(sources[0], 2, axis=1), 2, axis=2)
                elif node["op"] == "maxpool":
                    source = sources[0]
                    if node["kernel"] != 2 or node["stride"] != 2 or node["padding"] != 0:
                        raise ValueError("only the actual PAN 2x2 maxpool is supported")
                    channels, height, width = source.shape
                    value = source.reshape(channels, height // 2, 2, width // 2, 2).max(axis=(2, 4))
                else:
                    raise ValueError("unsupported integer primitive: " + node["op"])
                if value.dtype != np.int8 or tuple(value.shape) != tuple(node["output_shape"]):
                    raise ValueError("integer primitive dtype/geometry differs at " + node["id"])
                values[node["id"]] = value
            saved = values if retain else {path: values[path] for path in self.graph["outputs"].values()}
            for path, value in saved.items():
                batched_values.setdefault(path, []).append(value)
        stacked = {path: np.stack(values) for path, values in batched_values.items()}
        for node in self.graph["nodes"]:
            if node["id"] in stacked:
                value = stacked[node["id"]]
                traces.append(dict(node=node["id"], op=node["op"], dtype=str(value.dtype),
                                   element_count=int(value.size),
                                   integer_sha256=hashlib.sha256(value.tobytes()).hexdigest()))
        exponents = {node["id"]: node["output_exp"] for node in self.graph["nodes"]}
        return IntegerElasticResult(
            outputs={level: stacked[path] for level, path in self.graph["outputs"].items()},
            output_exponents={level: exponents[path] for level, path in self.graph["outputs"].items()},
            values=stacked if retain else {}, trace=traces)

    def parameter_package(self):
        return dict(numeric_contract="NUMERIC_CONTRACT_V1", numeric_contract_hash=EXPECTED_HASH,
                    graph=deepcopy(self.graph),
                    weights={path: torch.from_numpy(record["weight"].copy()) for path, record in self.parameters.items()},
                    biases={path: torch.from_numpy(record["bias"].copy()) for path, record in self.parameters.items()},
                    execution_kind=self.execution_kind,
                    hls_parity="NOT_RUN", board_parity="NOT_RUN", hardware_feasibility="UNKNOWN")

    def parameter_hash(self):
        return canonical_hash({path: dict(
            weight_sha256=hashlib.sha256(record["weight"].tobytes()).hexdigest(),
            bias_sha256=hashlib.sha256(record["bias"].tobytes()).hexdigest())
            for path, record in sorted(self.parameters.items())})


@dataclass
class PreparedElasticPTQ:
    folded_model: StaticElasticNeckDetector
    fake_model: FakeQuantElasticGraph
    integer_reference: IntegerElasticReference
    extrema: dict[str, Any]
    profile: dict[str, Any]


@torch.no_grad()
def prepare_elastic_ptq(model: StaticElasticNeckDetector, calibration: TrainCalibration,
                        *, device="cpu", profile="W8A8") -> PreparedElasticPTQ:
    if profile not in SUPPORTED_PROFILES:
        raise NotImplementedError("new graph currently verifies W8A8 only; head INT16 integer arithmetic/export is not implemented")
    if not isinstance(model, StaticElasticNeckDetector) or model.training:
        raise ValueError("PTQ requires an independent selected StaticElasticNeckDetector in eval mode")
    if not isinstance(calibration, TrainCalibration):
        raise ValueError("calibration requires source-bound TrainCalibration, never raw valid/test tensors")
    calibration.verify_manifest()
    load_numeric_contract_v1()
    if canonical_hash(integer_numeric.CONTRACT) != EXPECTED_HASH:
        raise ValueError("retained integer arithmetic declaration differs from NUMERIC_CONTRACT_V1")
    original_hash = model_state_hash(model)
    source_float_dtypes = sorted({str(value.dtype) for value in model.state_dict().values()
                                 if value.is_floating_point()})
    if source_float_dtypes != ["torch.float32"]:
        raise ValueError("registered FP64 BN reference requires an unchanged FP32 source checkpoint")
    # Promote the original stored FP32 coefficients before folding. This is a
    # separately registered mathematical reference, not a relaxed FP32 gate.
    folded = deepcopy(model).to(device=device, dtype=torch.float64).eval()
    pairs = fold_batch_norms_inplace(folded)
    if any(isinstance(module, nn.BatchNorm2d) for module in folded.modules()):
        raise ValueError("unfolded BN remains in the selected new graph")
    graph = build_cost_graph(model.candidate_architecture, num_classes=model.num_classes)
    _validate_folded_graph(folded, graph)
    statistics = {"image": dict(output_max_abs=0.0)}
    for node in graph["nodes"]:
        statistics[node["id"]] = dict(output_max_abs=0.0)
        if node["op"] == "conv":
            statistics[node["id"]]["input_max_abs"] = 0.0
        elif node["op"] in {"add", "residual_add"}:
            statistics[node["id"]].update(left_max_abs=0.0, right_max_abs=0.0)
    images_seen, first_witness = 0, None
    for images in calibration.batches(device):
        reference_images = images.to(dtype=torch.float64)
        values = _float_graph(folded, graph, reference_images)
        if first_witness is None:
            original_fp32 = deepcopy(model).to(device).eval()(images)["predictions"]
            observed_folded_fp32 = deepcopy(model).to(device).eval()
            fold_batch_norms_inplace(observed_folded_fp32)
            fp32_folded_predictions = observed_folded_fp32(images)["predictions"]
            fp32_comparison = _prediction_comparison(original_fp32, fp32_folded_predictions,
                                                     atol=1e-5, rtol=1e-5)
            original_reference = deepcopy(model).to(device=device, dtype=torch.float64).eval()
            original = original_reference(reference_images)["predictions"]
            actual_folded = folded(reference_images)["predictions"]
            graph_predictions = tuple(values[graph["outputs"][level]] for level in LEVELS)
            bn_errors = _prediction_errors(original, actual_folded)
            graph_errors = _prediction_errors(actual_folded, graph_predictions)
            mathematical_comparison = _prediction_comparison(original, actual_folded, atol=1e-5, rtol=1e-5)
            tight_comparison = _prediction_comparison(original, actual_folded, atol=1e-8, rtol=1e-8)
            if not mathematical_comparison["all_heads_allclose"] or not tight_comparison["all_heads_allclose"]:
                raise ValueError("paired FP64 mathematical BN-folding prediction parity failed")
            if any(error != 0 for error in graph_errors):
                raise ValueError("explicit new graph differs numerically from actual folded TD/PAN forward")
            first_witness = dict(
                batch_images=int(images.shape[0]), images_sha256=_tensor_sha(images),
                reference_images_sha256=_tensor_sha(reference_images),
                source_image_dtype=str(images.dtype), reference_image_dtype=str(reference_images.dtype),
                paired_original_fp64_state_sha256=model_state_hash(original_reference),
                bn_folded_prediction_max_errors=bn_errors,
                true_new_graph_prediction_max_errors=graph_errors,
                comparison="paired_FP64_original_static_vs_FP64_folded_static_vs_explicit_new_DAG",
                bn_fold_atol=1e-5, bn_fold_rtol=1e-5,
                mathematical_fp64=mathematical_comparison,
                tighter_mathematical_fp64=tight_comparison,
                observed_source_fp32_vs_folded_fp32=fp32_comparison,
                observed_source_fp32_vs_folded_fp64=_prediction_comparison(
                    original_fp32, actual_folded, atol=1e-5, rtol=1e-5),
                fp32_observation_is_acceptance_gate=False,
                graph_exact_equality=True, witness_scope="first_train_calibration_batch",
            )
            del original_reference, observed_folded_fp32, original, original_fp32, fp32_folded_predictions
        for path, value in values.items():
            maximum = float(value.abs().max())
            if not np.isfinite(maximum):
                raise ValueError("non-finite train observation at " + path)
            statistics[path]["output_max_abs"] = max(statistics[path]["output_max_abs"], maximum)
        for node in graph["nodes"]:
            if node["op"] == "conv":
                observed = statistics[node["id"]]
                observed["input_max_abs"] = max(observed["input_max_abs"],
                                                float(values[node["inputs"][0]].abs().max()))
            elif node["op"] in {"add", "residual_add"}:
                observed = statistics[node["id"]]
                observed["left_max_abs"] = max(observed["left_max_abs"],
                                               float(values[node["inputs"][0]].abs().max()))
                observed["right_max_abs"] = max(observed["right_max_abs"],
                                                float(values[node["inputs"][1]].abs().max()))
        images_seen += int(images.shape[0])
    if images_seen != calibration.manifest["number_of_calibration_images"] or first_witness is None:
        raise ValueError("calibration image count differs from the bound train prefix")
    if model_state_hash(model) != original_hash:
        raise ValueError("source network changed during PTQ preparation")
    exponents = {"image": exponent_from_maxabs(statistics["image"]["output_max_abs"])}
    numeric_graph = deepcopy(graph)
    numeric_graph["input_exp"] = exponents["image"]
    numeric_graph["numeric_contract_hash"] = EXPECTED_HASH
    numeric_graph["graph_scope"] = "explicit_new_software_integer_topology_not_hardware_export"
    numeric_graph["quantization_status"] = "W8A8_SOFTWARE_REFERENCE_PREPARED_PARITY_NOT_YET_RUN"
    layers, adds = [], []
    for node in numeric_graph["nodes"]:
        source_exps = [exponents[source] for source in node["inputs"]]
        output_exp = (source_exps[0] if node["op"] in {"nearest", "maxpool"}
                      else exponent_from_maxabs(statistics[node["id"]]["output_max_abs"]))
        node["output_exp"] = exponents[node["id"]] = output_exp
        if node["op"] == "conv":
            conv = folded.get_submodule(node["id"])
            _, weight_exps = quantize_weight(conv.weight, granularity="per_tensor", bits=8)
            multiplier, shift = ratio_from_exponents(source_exps[0] + weight_exps[0], output_exp)
            node.update(input_exp=source_exps[0], weight_exp=weight_exps[0],
                        multiplier=multiplier, shift=shift, q6=integer_numeric.q6(output_exp))
            layers.append(dict(name=node["id"], input_exp=node["input_exp"], weight_exp=node["weight_exp"],
                               output_exp=output_exp, multiplier=multiplier, shift=shift,
                               input_bits=8, weight_bits=8, output_bits=8, activation=node["activation"],
                               groups=node["groups"], kernel=node["kernel"],
                               folded_weight_sha256=_tensor_sha(conv.weight),
                               folded_bias_sha256=_tensor_sha(conv.bias)))
        elif node["op"] in {"add", "residual_add"}:
            common = max(source_exps)
            # Validate the exact retained H2 common-scale and output transitions.
            for source_exp in source_exps:
                ratio_from_exponents(source_exp, common)
            ratio_from_exponents(common, output_exp)
            node.update(input_exp_a=source_exps[0], input_exp_b=source_exps[1],
                        common_exp=common, alignment="H2_INT8_align_each_INT32_sum_INT8_requant")
            adds.append(dict(name=node["id"], op=node["op"], inputs=list(node["inputs"]),
                             input_exponents=source_exps, common_exp=common, output_exp=output_exp,
                             observation=deepcopy(statistics[node["id"]])))
        else:
            node["input_exp"] = source_exps[0]
    numeric_graph["numeric_graph_sha256"] = canonical_hash(numeric_graph)
    for comparison in (first_witness["observed_source_fp32_vs_folded_fp32"],
                       first_witness["observed_source_fp32_vs_folded_fp64"]):
        for row in comparison["heads"]:
            lsb = power_of_two(exponents[graph["outputs"][row["level"]]])
            row.update(train64_calibrated_output_lsb=lsb, max_abs_error_in_output_lsb=row["max_abs_error"] / lsb)
    fake = FakeQuantElasticGraph(folded, numeric_graph).to(device).eval()
    integer = IntegerElasticReference(fake)
    source_paths = (Path(__file__), Path(integer_numeric.__file__))
    profile_record = dict(
        adapter_version=ADAPTER_VERSION, precision_profile="W8A8", supported_profiles=list(SUPPORTED_PROFILES),
        unsupported_profiles=["HEAD_W8A16", "HEAD_W16A16"],
        numeric_contract="NUMERIC_CONTRACT_V1", numeric_contract_hash=EXPECTED_HASH,
        bitwidth=dict(activation=8, weight=8, accumulator=32, requant_product=64),
        rounding="nearest_even", signed_range=[-128, 127], zero_point=0,
        overflow="twos_complement_modulo_2pow32", weight_granularity="per_tensor",
        architecture=deepcopy(model.candidate_architecture), num_classes=model.num_classes,
        original_model_state_sha256=original_hash, folded_model_state_sha256=model_state_hash(folded),
        source_checkpoint_floating_dtypes=source_float_dtypes,
        folded_reference_floating_dtypes=sorted({str(value.dtype) for value in folded.state_dict().values()
                                                if value.is_floating_point()}),
        activation_observation_dtype="torch.float64",
        weight_quantization_preparation=dict(
            exponent_maximum_dtype="torch.float64", nearest_even_weight_rounding_dtype="torch.float32",
            bias_rounding_dtype="torch.float64", implementation="retained_H2_quantize_weight_and_quantize_bias"),
        cost_graph_sha256=graph["cost_graph_sha256"], numeric_graph_sha256=numeric_graph["numeric_graph_sha256"],
        calibration=deepcopy(calibration.manifest), layers=layers, adds=adds,
        bn_folding=dict(folded_pair_count=len(pairs), pairs=pairs, witness=first_witness,
                        reference_kind="FP64_mathematical_paired_BN_fold_from_unchanged_FP32_checkpoint",
                        coefficient_dtype="torch.float64", registered_gate_atol=1e-8, registered_gate_rtol=1e-8,
                        original_fp32_gate_atol=1e-5, original_fp32_gate_rtol=1e-5,
                        original_fp32_tolerance_relaxed=False),
        input_exponent=exponents["image"], output_exponents={level: exponents[path] for level, path in graph["outputs"].items()},
        fake_execution_kind=fake.execution_kind, integer_execution_kind=integer.execution_kind,
        fake_model_state_sha256=model_state_hash(fake), integer_parameters_sha256=integer.parameter_hash(),
        adapter_sources_sha256={str(path.resolve()): _file_sha(path) for path in source_paths},
        new_neck_residual_and_PAN_adds_quantized=True,
        integer_vs_fake_parity="NOT_RUN", integer_vs_HLS_parity="NOT_RUN", board_parity="NOT_RUN",
        quantized_accuracy_retention="NOT_EVALUATED", hardware_feasibility="UNKNOWN",
        full_hardware_export="NOT_IMPLEMENTED_FOR_NEW_GRAPH", formal_eligible=False,
        test_accessed=False, test_evaluated=False,
    )
    return PreparedElasticPTQ(folded, fake, integer, statistics, profile_record)


@torch.no_grad()
def verify_fake_integer_parity(prepared: PreparedElasticPTQ, images: torch.Tensor):
    """Measure all graph-node mismatches; leave hardware evidence unclaimed."""
    fake = prepared.fake_model
    if (model_state_hash(fake) != prepared.profile["fake_model_state_sha256"]
            or prepared.integer_reference.parameter_hash() != prepared.profile["integer_parameters_sha256"]):
        raise ValueError("PTQ parameter drift before parity measurement")
    parameter = next(fake.buffers())
    fake_values = fake.forward_all(images.to(parameter.device))
    integer = prepared.integer_reference
    result = integer.forward(integer.quantize_input(images), retain=True)
    exponents = {"image": integer.graph["input_exp"]}
    exponents.update({node["id"]: node["output_exp"] for node in integer.graph["nodes"]})
    rows = []
    for path, expected in result.values.items():
        simulated = integer_numeric.quantize_real(fake_values[path].cpu().numpy(), exponents[path])
        mismatch = int(np.count_nonzero(expected != simulated))
        rows.append(dict(node=path, element_count=int(expected.size), mismatch_count=mismatch,
                         integer_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                         fake_quantized_sha256=hashlib.sha256(simulated.tobytes()).hexdigest()))
    total = sum(row["mismatch_count"] for row in rows)
    return dict(status="PASS_SOFTWARE_INTEGER_FAKE_PARITY" if total == 0 else "FAIL_SOFTWARE_INTEGER_FAKE_PARITY",
                scope="independent_numpy_integer_vs_H2_float_fake_on_bound_new_DAG",
                input_images_sha256=_tensor_sha(images), input_images=int(images.shape[0]),
                numeric_graph_sha256=integer.graph["numeric_graph_sha256"],
                calibration_sha256=prepared.profile["calibration"]["calibration_sha256"],
                original_model_state_sha256=prepared.profile["original_model_state_sha256"],
                integer_parameters_sha256=prepared.profile["integer_parameters_sha256"],
                mismatches=total, rows=rows, formal_eligible=False,
                integer_vs_HLS_parity="NOT_RUN", board_parity="NOT_RUN", test_evaluated=False)


def save_ptq_artifacts(output, prepared: PreparedElasticPTQ, *, witness=None):
    folder = Path(output)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("PTQ artifacts require a new empty directory")
    folder.mkdir(parents=True, exist_ok=True)
    for filename, payload in (("profile.json", prepared.profile), ("extrema.json", prepared.extrema),
                              ("numeric_graph.json", prepared.integer_reference.graph)):
        (folder / filename).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    torch.save(prepared.integer_reference.parameter_package(), folder / "integer_parameters.pt")
    if witness is not None:
        (folder / "software_parity_witness.json").write_text(
            json.dumps(witness, indent=2, allow_nan=False), encoding="utf-8")
    paths = [path for path in folder.iterdir() if path.is_file()]
    (folder / "artifact_manifest.json").write_text(json.dumps(
        dict(files={path.name: _file_sha(path) for path in paths}, formal_eligible=False,
             hardware_export="NOT_IMPLEMENTED_FOR_NEW_GRAPH", test_evaluated=False), indent=2), encoding="utf-8")
    return folder


__all__ = ["ADAPTER_VERSION", "SUPPORTED_PROFILES", "TrainCalibration", "model_state_hash",
           "PreparedElasticPTQ", "prepare_elastic_ptq", "ExactValuedH2FakeQuantConv", "FakeQuantElasticGraph",
           "IntegerElasticReference", "IntegerElasticResult", "verify_fake_integer_parity", "save_ptq_artifacts"]
