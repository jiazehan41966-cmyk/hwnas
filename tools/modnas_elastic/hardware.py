"""Honest hardware boundary for the NEW 72-candidate ELASTIC/PAN space.

Old B1 HLS evidence is reused only for exactly identical primitives. Missing
costs remain null. The differentiable hardware objective requires a complete,
source-bound NEW-space table; absent that table hardware search is blocked.
The separate Params objective is software-only and must never be called latency
or LUT. Finite-table expectation replaces, rather than reproduces, the author's
multi-device meta-predictor.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import torch
from torch import nn

from .cost_graph import (ROOT, DEFAULT_PRECISION, build_cost_graph, canonical_hash,
                         h3_shape, primitive_shape, validate_architecture)

BLOCKED = "BLOCKED_MISSING_NEW_SPACE_COST"
METRICS = ("Latency_us", "LUT", "DSP", "BRAM18")
MAPPING_POLICY = "reference_min_latency"
PART = "xczu7ev-ffvc1156-2-e"
H3_ROWS = ROOT / "artifacts/fpga_h3/20260918_batch01/clean.json"
H3_MANIFEST = ROOT / "reports/fpga_h3_cost_db/H3_DATABASE_MANIFEST.json"
_VALIDATED_TABLE = object()


def enumerate_architectures() -> list[dict[str, Any]]:
    # Keep no dependency on model construction or old-space enumeration.
    top = list(itertools.product((3, 5), repeat=3))
    return ([dict(pan=False, td_kernels=list(td), pan_kernels=[]) for td in top]
            + [dict(pan=True, td_kernels=list(td), pan_kernels=list(bottom))
               for td in top for bottom in itertools.product((3, 5), repeat=3)])


def _space(space: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    expected = enumerate_architectures()
    actual = expected if space is None else [validate_architecture(arch) for arch in space]
    if actual != expected:
        raise ValueError("Cost table requires all 72 NEW architectures in canonical E0-then-E1 order")
    return actual


def _weights(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating Tensor with shape {shape}")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()) or bool((detached < 0).any()):
        raise ValueError(f"{name} must contain finite nonnegative probabilities")
    if not torch.allclose(detached.sum(-1), torch.ones_like(detached.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError(f"{name} must sum to one along its last dimension")
    return value


def architecture_probabilities(weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Canonical 72 probabilities, merging all inactive PAN kernel assignments.

    Inputs are already softmax probabilities, not logits. PAN-off probabilities
    do not depend on pan_kernel_weights, eliminating inactive-tail bias.
    """
    pan = _weights(weights["pan_weights"], (2,), "pan_weights")
    td = _weights(weights["td_kernel_weights"], (3, 2), "td_kernel_weights")
    bottom = _weights(weights["pan_kernel_weights"], (3, 2), "pan_kernel_weights")
    if pan.device != td.device or pan.device != bottom.device:
        raise ValueError("Architecture weights must share a device")
    if pan.dtype != td.dtype or pan.dtype != bottom.dtype:
        raise ValueError("Architecture weights must share a dtype")
    td_joint = torch.stack([td[0, i] * td[1, j] * td[2, k]
                            for i, j, k in itertools.product(range(2), repeat=3)])
    pan_joint = torch.stack([bottom[0, i] * bottom[1, j] * bottom[2, k]
                             for i, j, k in itertools.product(range(2), repeat=3)])
    probabilities = torch.cat((pan[0] * td_joint, (pan[1] * td_joint[:, None] * pan_joint[None, :]).reshape(-1)))
    # Normalize numerical roundoff while preserving gradients and canonical mass.
    return probabilities / probabilities.sum()


def _probabilities(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.shape[-1:] != (72,):
        raise ValueError("Expected floating probabilities [...,72]")
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()) or bool((detached < 0).any()):
        raise ValueError("Candidate probabilities must be finite and nonnegative")
    if not torch.allclose(detached.sum(-1), torch.ones_like(detached.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError("Candidate probabilities must sum to one")
    return value


class DifferentiableParamsTable(nn.Module):
    """Exact full-model software Parameters expectation, including BN and bias."""
    objective_name = "Params"
    measurement_scope = "SOFTWARE_PARAMETER_COUNT_NOT_HARDWARE"
    hardware_search_status = BLOCKED

    def __init__(self, space=None, *, num_classes: int = 2):
        super().__init__()
        self.architectures = _space(space)
        self.num_classes = num_classes
        self.register_buffer("values", torch.tensor([build_cost_graph(arch, num_classes=num_classes)["Params"]
                                                     for arch in self.architectures], dtype=torch.float64))

    def forward(self, probabilities: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(probabilities, Mapping):
            probabilities = architecture_probabilities(probabilities)
        probabilities = _probabilities(probabilities)
        return probabilities @ self.values.to(device=probabilities.device, dtype=probabilities.dtype)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Evidence file path is required")
    path = Path(value)
    return path if path.is_absolute() else root / path


def _bound_file(root: Path, path: str, digest: str) -> Path:
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Evidence SHA256 is required")
    source = _file(root, path)
    if not source.is_file() or _sha(source) != digest:
        raise ValueError(f"Missing or hash-mismatched evidence: {path}")
    return source


def _xml_metrics(path: Path) -> dict[str, float]:
    tree = ET.parse(path).getroot()
    if (tree.findtext(".//UserAssignments/Part") != PART
            or tree.findtext(".//ReportVersion/Version") != "2022.2"
            or float(tree.findtext(".//UserAssignments/TargetClockPeriod")) != 5):
        raise ValueError("CSYNTH XML platform/tool/clock mismatch")
    def number(tag):
        value = float(tree.findtext(tag))
        if not math.isfinite(value) or value < 0:
            raise ValueError("Nonfinite or negative CSYNTH metric")
        return value
    result = {metric: number(".//AreaEstimates/Resources/" + tag)
              for metric, tag in (("LUT", "LUT"), ("DSP", "DSP"), ("BRAM18", "BRAM_18K"))}
    result.update(cycles=number(".//SummaryOfOverallLatency/Worst-caseLatency"),
                  estimated_clock_ns=number(".//SummaryOfTimingAnalysis/EstimatedClockPeriod"))
    return result


def _verify_h3_row(row: Mapping[str, Any], hashes: Mapping[str, str]) -> list[str]:
    """Validate retained primitive source/report/witness provenance without HLS."""
    errors = []
    try:
        expected = dict(status="VALID", LEGAL_MAPPING_CHECK="PASS", csim_status="PASS_BIT_EXACT",
                        csynth_status="PASS", tool_stage="CSYNTH", part=PART,
                        part_reported=PART, tool_version="2022.2", tool_reported="2022.2")
        for key, value in {**expected, **hashes, **DEFAULT_PRECISION}.items():
            if row.get(key) != value:
                errors.append(key + "_mismatch")
        if float(row.get("target_clock_ns", -1)) != 5 or float(row.get("clock_reported", -1)) != 5:
            errors.append("target_clock_mismatch")
        report = _bound_file(ROOT, row["source_report_path"], row["source_report_sha256"])
        actual = _xml_metrics(report)
        for metric in ("LUT", "DSP", "BRAM18", "cycles", "estimated_clock_ns"):
            if float(row[metric]) != actual[metric]:
                errors.append("report_" + metric + "_mismatch")
        if float(row["latency_us"]) != actual["cycles"] * 5 / 1000:
            errors.append("report_latency_mismatch")
        if actual["estimated_clock_ns"] > 5 or row.get("timing_status") != "ESTIMATED_TARGET_MET":
            errors.append("timing_not_eligible")
        if row.get("tool_log_path"):
            _bound_file(ROOT, row["tool_log_path"], row["tool_log_sha256"])
        case = _file(ROOT, row["case_dir"])
        key = json.loads((case / "measurement_key.json").read_text(encoding="utf-8"))
        source_files = key["source_files"]
        if not source_files or key.get("source_hash") != row["source_hash"] or canonical_hash(source_files) != row["source_hash"]:
            errors.append("measurement_key_source_hash_mismatch")
        for name, digest in source_files.items():
            _bound_file(case, name, digest)
        for name, digest in (key.get("witness_files") or {}).items():
            _bound_file(case, name, digest)
        _bound_file(case, "reference.bin", row["reference_hash"])
        _bound_file(case, "input.bin", row["input_hash"])
        _bound_file(report.parent, "hardware_output.bin", row["reference_hash"])
        parameter_hashes = {file.name: _sha(file) for file in case.glob("*.bin")
                            if file.name not in ("input.bin", "reference.bin")}
        if canonical_hash(parameter_hashes) != row.get("binary_parameters_hash"):
            errors.append("binary_parameters_hash_mismatch")
        # Match the original H3 collector's canonical implementation, preserving
        # its submitted SIMD separately rather than inventing SIMD aliases.
        canonical_key = dict(key)
        if key.get("kind") in ("mb", "dw", "up"):
            canonical_key.update(SIMD=1, input_tile=1)
        for field in (*DEFAULT_PRECISION, "shape_id", "operator_type", "Cin", "Cout", "H", "W",
                      "kernel", "stride", "groups", "activation", "PE", "input_tile", "output_tile",
                      "buffer_profile", "part", "target_clock_ns"):
            if field in canonical_key and row.get(field) != canonical_key[field]:
                errors.append("measurement_key_" + field + "_mismatch")
        # Historical DW requests SIMD2 were canonically implemented with SIMD1.
        if int(row["Cin"]) % int(row["SIMD"]) or int(row["Cout"]) % int(row["PE"]):
            errors.append("illegal_channel_mapping")
    except (ValueError, KeyError, OSError, TypeError, ET.ParseError) as error:
        errors.append("retained_evidence_invalid:" + str(error))
    return errors


class PrimitiveEvidenceCache:
    """One-time exact H3 primitive admission and cached NEW graph gap queries."""
    def __init__(self):
        self.manifest = json.loads(H3_MANIFEST.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "H3_DB_READY":
            raise ValueError("H3 source manifest is not ready")
        self.hashes = {key: self.manifest[key] for key in ("numeric_contract_hash", "interface_contract_hash")}
        for name in ("numeric", "interface"):
            contract = json.loads((ROOT / f"reports/fpga_h2_contract/{name}_contract_v1.json").read_text(encoding="utf-8"))
            if canonical_hash(contract) != self.hashes[name + "_contract_hash"]:
                raise ValueError("H3 source contract hash mismatch")
        self.rows = json.loads(H3_ROWS.read_text(encoding="utf-8"))
        self.cache, self.admissions = {}, {}
        self.counts = Counter(source_library_loads=1)

    def query(self, node: Mapping[str, Any]) -> dict[str, Any]:
        self.counts["total_node_queries"] += 1
        key = node["primitive_shape_sha256"]
        if key in self.cache:
            self.counts["cache_hits"] += 1
            return deepcopy(self.cache[key])
        self.counts["distinct_primitive_queries"] += 1
        shape = h3_shape(node)
        result = dict(status="MISSING", primitive_shape_sha256=key,
                      reason="new_operator_has_no_exact_cost_evidence", measurement=None)
        if shape is not None:
            self.counts["source_shape_queries"] += 1
            exact = [row for row in self.rows if all(row.get(field) == value for field, value in shape.items())
                     and all(row.get(field) == value for field, value in DEFAULT_PRECISION.items())]
            eligible = []
            errors = {}
            for row in exact:
                point = row["point_id"]
                if point not in self.admissions:
                    self.admissions[point] = _verify_h3_row(row, self.hashes)
                    self.counts["source_rows_validated"] += 1
                if self.admissions[point]:
                    errors[point] = self.admissions[point]
                else:
                    eligible.append(row)
            if eligible:
                selected = min(eligible, key=lambda row: (row["latency_us"], row["DSP"], row["point_id"]))
                fields = ("point_id", "shape_id", "source_hash", "source_report_path", "source_report_sha256",
                          "tool_log_path", "tool_log_sha256", "case_dir", "PE", "SIMD", "input_tile", "output_tile",
                          "spatial_tile", "buffer_profile", "part", "tool_version", "target_clock_ns",
                          "numeric_contract_hash", "interface_contract_hash", "latency_us", "LUT", "DSP", "BRAM18")
                result.update(status="EXACT_PRIMITIVE_HLS_EVIDENCE", reason=None,
                              measurement={field: selected[field] for field in fields},
                              evidence_scope="inherited_B1_primitive_only_not_new_graph_integer_or_deployment_evidence")
                self.counts["distinct_exact_primitive_hits"] += 1
            else:
                result.update(reason="no_exact_shape_precision_eligible_source_row", admission_errors=errors)
        if result["status"] == "MISSING":
            self.counts["distinct_missing_primitives"] += 1
        self.cache[key] = result
        return deepcopy(result)


def build_hardware_gap_report(space=None, *, num_classes: int = 2) -> dict[str, Any]:
    architectures = _space(space)
    database = PrimitiveEvidenceCache()
    rows, gaps = [], {}
    for arch in architectures:
        graph = build_cost_graph(arch, num_classes=num_classes)
        queries = [(node, database.query(node)) for node in graph["nodes"]]
        exact = [dict(node_id=node["id"], **query) for node, query in queries if query["status"] != "MISSING"]
        missing = [(node, query) for node, query in queries if query["status"] == "MISSING"]
        for node, query in missing:
            key = node["primitive_shape_sha256"]
            gap = gaps.setdefault(key, dict(primitive_shape_sha256=key, primitive=primitive_shape(node),
                                           precision=deepcopy(DEFAULT_PRECISION), reason=query["reason"],
                                           mapping_policy=MAPPING_POLICY, part=PART,
                                           tool_version="2022.2", target_clock_ns=5,
                                           occurrence_count=0, architecture_sha256s=[], node_ids=[],
                                           required_evidence=["frozen_quantization_and_operator_contract",
                                                              "source_bound_BIT_EXACT_integer_witness",
                                                              "source_bound_valid_timing_eligible_CSYNTH_report"],
                                           task_status="NOT_RUN_NO_SYNTHESIS_REQUESTED"))
            gap["occurrence_count"] += 1
            if graph["architecture_sha256"] not in gap["architecture_sha256s"]:
                gap["architecture_sha256s"].append(graph["architecture_sha256"])
            if node["id"] not in gap["node_ids"]:
                gap["node_ids"].append(node["id"])
        rows.append(dict(architecture=arch, architecture_sha256=graph["architecture_sha256"],
                         cost_graph_sha256=graph["cost_graph_sha256"], Params=graph["Params"],
                         total_primitive_nodes=len(queries), exact_primitive_nodes=len(exact),
                         missing_primitive_nodes=len(missing), exact_measurements=exact,
                         missing_primitive_sha256s=[node["primitive_shape_sha256"] for node, _ in missing],
                         status=BLOCKED, **{metric: None for metric in METRICS}))
    return dict(schema_version=1, space="NEW_fixed_B1_four_scale_U48_E3_TD_kernel3or5_optional_PAN_kernel3or5",
                candidate_count=72, space_sha256=canonical_hash(architectures), num_classes=num_classes,
                precision=deepcopy(DEFAULT_PRECISION), mapping_policy=MAPPING_POLICY,
                hardware_search_status=BLOCKED, complete_hardware_labels=0,
                adapter_source_sha256={str(source.relative_to(ROOT)).replace("\\", "/"): _sha(source)
                                       for source in (Path(__file__), Path(__file__).with_name("cost_graph.py"),
                                                      ROOT / "tools/modnas_elastic/model.py",
                                                      ROOT / "configs/detector_elastic_pilot_model.yaml")
                                       if source.is_file()},
                source_library=dict(path=str(H3_ROWS), sha256=_sha(H3_ROWS), manifest_path=str(H3_MANIFEST),
                                    manifest_sha256=_sha(H3_MANIFEST), measurement_scope="OP_LEVEL_HLS",
                                    **database.hashes),
                latency_scope="HLS_CSYNTH_serial_operator_sum_not_board_e2e",
                resource_scope="kernel_only_peak_operator_not_integrated_post_route_excludes_shell_DDR",
                shell_reservation="UNKNOWN", lookup_accounting=dict(database.counts),
                accounting_note="One source library load; total_node_queries includes repeated primitives; "
                                "distinct_primitive_queries is cache-miss denominator, not candidate count; "
                                "72 candidates here are structural audit, zero trained/evaluated candidates.",
                candidates=rows, missing_measurement_tasks=list(gaps.values()),
                next_gate="complete_new_integer_contract_and_source_bound_72_cost_table",
                no_test_images_accessed=True, no_training=True, no_hls_or_board_execution=True)


class ExternalMeasuredCostTable(nn.Module):
    """Complete NEW-space OP_LEVEL_HLS labels with retained per-node proofs.

    JSON schema: schema_version, space_sha256, num_classes, precision,
    mapping_policy, measurement_scope=OP_LEVEL_HLS, quant_contract={path,sha256},
    rows (exactly 72). Each row binds architecture_sha256/cost_graph_sha256 and
    measurements (one per graph node) plus Latency_us/LUT/DSP/BRAM18. Each
    measurement binds primitive, numeric_contract_hash, interface_contract_hash,
    PE/SIMD/buffer_profile/platform, source_files and source_hash,
    source_report_path/sha256, csim_reference_path/sha256 and output equivalents.
    Latency is the serial sum; resources are operator peaks. The loader checks
    report metrics, source bytes, witness parity and complete graph coverage.
    Relative evidence paths resolve against the table's directory.
    """
    hardware_search_status = "READY_COMPLETE_NEW_SPACE_OP_LEVEL_HLS_TABLE"
    measurement_scope = "OP_LEVEL_HLS"
    adaptation = "finite_table_expectation_engineering_replacement_not_author_multi_device_meta_predictor"

    def __init__(self, values: torch.Tensor, metadata: Mapping[str, Any], *, _proof=None):
        super().__init__()
        if _proof is not _VALIDATED_TABLE:
            raise ValueError("Use from_json: hardware values require verified NEW-space evidence")
        if values.shape != (72, 4) or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError("Hardware labels require finite nonnegative [72,4] values")
        self.register_buffer("values", values.to(torch.float64))
        self.metadata = deepcopy(dict(metadata))

    @classmethod
    def from_json(cls, path: str | Path, space=None) -> "ExternalMeasuredCostTable":
        path = Path(path)
        table = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
        architectures = _space(space)
        space_hash = canonical_hash(architectures)
        expected = dict(schema_version=1, space_sha256=space_hash, precision=DEFAULT_PRECISION,
                        mapping_policy=MAPPING_POLICY, measurement_scope="OP_LEVEL_HLS")
        for key, value in expected.items():
            if table.get(key) != value:
                raise ValueError(f"Cost table {key} mismatch")
        contract_record = table.get("quant_contract", {})
        contract_path = _bound_file(base, contract_record.get("path"), contract_record.get("sha256"))
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for key, value in dict(status="NEW_ELASTIC_INTEGER_CONTRACT_VERIFIED", space_sha256=space_hash,
                               precision=DEFAULT_PRECISION, num_classes=table.get("num_classes")).items():
            if contract.get(key) != value:
                raise ValueError(f"New integer contract {key} mismatch")
        if not all(isinstance(contract.get(key), str) and len(contract[key]) == 64
                   for key in ("numeric_contract_hash", "interface_contract_hash")):
            raise ValueError("New integer numeric/interface contract hash required")
        rows = table.get("rows")
        if not isinstance(rows, list) or len(rows) != 72:
            raise ValueError("Complete hardware table must contain all 72 NEW candidates")
        values = []
        for arch, row in zip(architectures, rows):
            graph = build_cost_graph(arch, num_classes=table["num_classes"])
            for key in ("architecture_sha256", "cost_graph_sha256"):
                if row.get(key) != graph[key]:
                    raise ValueError(f"Candidate {key} mismatch")
            measurements = row.get("measurements")
            if not isinstance(measurements, list) or len(measurements) != len(graph["nodes"]):
                raise ValueError("Every NEW graph primitive requires measured evidence")
            by_node = {measurement.get("node_id"): measurement for measurement in measurements}
            if len(by_node) != len(measurements) or set(by_node) != {node["id"] for node in graph["nodes"]}:
                raise ValueError("Missing or duplicated primitive measurement")
            metrics = [_verify_external_node(by_node[node["id"]], node, contract, base) for node in graph["nodes"]]
            computed = dict(Latency_us=sum(metric["Latency_us"] for metric in metrics),
                            **{name: max(metric[name] for metric in metrics) for name in METRICS[1:]})
            for name in METRICS:
                if type(row.get(name)) not in (int, float) or not math.isfinite(row[name]) or row[name] != computed[name]:
                    raise ValueError(f"Aggregate {name} differs from measured primitive evidence")
            values.append([computed[name] for name in METRICS])
        metadata = {key: table[key] for key in expected}
        metadata.update(table_path=str(path.resolve()), table_sha256=_sha(path),
                        num_classes=table["num_classes"], quant_contract=contract_record,
                        shell_reservation="UNKNOWN", board_deployment="NOT_CLAIMED",
                        latency_scope="serial_operator_CSYNTH_not_board_e2e", resource_scope="peak_operator_not_whole_network",
                        adaptation=cls.adaptation)
        return cls(torch.tensor(values, dtype=torch.float64), metadata, _proof=_VALIDATED_TABLE)

    def forward(self, probabilities: torch.Tensor | Mapping[str, torch.Tensor],
                budgets: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
        if isinstance(probabilities, Mapping):
            probabilities = architecture_probabilities(probabilities)
        probabilities = _probabilities(probabilities)
        expected = probabilities @ self.values.to(device=probabilities.device, dtype=probabilities.dtype)
        result = {name: expected[..., index] for index, name in enumerate(METRICS)}
        if budgets is not None:
            if set(budgets) != set(METRICS) or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                                                 for value in budgets.values()):
                raise ValueError("Budgets require positive finite values for all four cost metrics")
            result = {name: result[name] / budgets[name] for name in METRICS}
        return result

    def exact(self, architecture: Mapping[str, Any]) -> dict[str, float]:
        arch = validate_architecture(architecture)
        index = enumerate_architectures().index(arch)
        return {name: float(self.values[index, column]) for column, name in enumerate(METRICS)}


def _verify_external_node(measurement, node, contract, base):
    if measurement.get("primitive") != primitive_shape(node):
        raise ValueError("Primitive shape/operator/activation mismatch")
    for key, value in dict(primitive_shape_sha256=node["primitive_shape_sha256"], precision=DEFAULT_PRECISION,
                           part=PART, tool_version="2022.2", target_clock_ns=5,
                           csim_status="PASS_BIT_EXACT", csynth_status="PASS",
                           numeric_contract_hash=contract["numeric_contract_hash"],
                           interface_contract_hash=contract["interface_contract_hash"]).items():
        if measurement.get(key) != value:
            raise ValueError(f"Primitive evidence {key} mismatch")
    if type(measurement.get("PE")) is not int or type(measurement.get("SIMD")) is not int or min(measurement["PE"], measurement["SIMD"]) < 1:
        raise ValueError("Exact hardware mapping PE/SIMD required")
    if not isinstance(measurement.get("buffer_profile"), str) or not measurement["buffer_profile"]:
        raise ValueError("Exact hardware buffer mapping required")
    files = measurement.get("source_files")
    if not isinstance(files, dict) or not files or canonical_hash(files) != measurement.get("source_hash"):
        raise ValueError("Source files and source hash are not bound")
    for source, digest in files.items():
        _bound_file(base, source, digest)
    reference = _bound_file(base, measurement["csim_reference_path"], measurement["csim_reference_sha256"])
    output = _bound_file(base, measurement["csim_output_path"], measurement["csim_output_sha256"])
    if reference.stat().st_size == 0 or _sha(reference) != _sha(output):
        raise ValueError("Integer witness parity failed")
    report = _bound_file(base, measurement["source_report_path"], measurement["source_report_sha256"])
    actual = _xml_metrics(report)
    if actual["estimated_clock_ns"] > 5:
        raise ValueError("CSYNTH report is not timing eligible")
    values = dict(Latency_us=actual["cycles"] * 5 / 1000,
                  **{name: actual[name] for name in METRICS[1:]})
    for name, value in values.items():
        if measurement.get(name) != value:
            raise ValueError(f"Primitive {name} mismatch with CSYNTH report")
    return values


__all__ = ["BLOCKED", "METRICS", "architecture_probabilities", "enumerate_architectures",
           "DifferentiableParamsTable", "PrimitiveEvidenceCache", "build_hardware_gap_report",
           "ExternalMeasuredCostTable"]
