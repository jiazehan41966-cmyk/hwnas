"""Source-bound hardware costs for the 192-v2 chain, with no hardware actions.

The T6 adapter replays existing local board evidence, preserving its ineligible
status.  A separate receipt interface consumes future protocol-admitted board
measurements.  Diagnostic observations never become formal measurements by
changing a flag on an imported row.  Network estimates remain serial operator
sums / peak standalone-shell resources, not integrated network measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SPACE = "sonar_backbone_neck192_v2"
PROFILES = ("W8A8", "HEAD_W8A16", "HEAD_W16A16")
POLICIES = ("reference_low", "reference_high", "reference_min_latency",
            "reference_min_dsp", "reference_min_lut", "reference_min_bram")
METRICS = ("compute_us", "transfer_us", "host_us", "LUT", "DSP", "BRAM18")
SHAPE_KEYS = ("op", "operator_type", "input_shapes", "output_shape", "Cin", "Cout", "H", "W",
              "kernel", "stride", "padding", "groups", "activation", "bias", "bn_folded",
              "interpolation", "alignment", "expanded_channels")
SEMANTIC_KEYS = ("dilation", "fixed_kernel", "fixed_kernel_denominator", "denominator", "dim",
                 "channel_order", "lut_sha256", "approximation", "quantization_axis")
_VERIFIED = object()
# Formal physical provenance is a reviewed trust anchor, not a self-signed
# receipt/review flag. Existing T6 observations remain diagnostic. Add entries
# only in a separately reviewed source revision after inspecting the real run.
REGISTERED_BOARD_OPERATOR_RECEIPTS = ()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bind(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def checked_file(binding: Mapping, base: str | Path = ROOT) -> Path:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise ValueError("Evidence requires an exact path/SHA256 binding")
    path = Path(binding["path"])
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve(strict=True)
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError("Evidence file missing or changed: " + str(path))
    return path


def require_registered_board_receipt(receipt_path, review_path) -> dict:
    key = dict(receipt_sha256=sha256_file(receipt_path), review_sha256=sha256_file(review_path))
    if not any(all(row.get(name) == value for name, value in key.items())
               for row in REGISTERED_BOARD_OPERATOR_RECEIPTS):
        raise ValueError("Unregistered formal board operator receipt: independently reviewed receipt/review hashes required")
    return key


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def primitive_shape(node: Mapping) -> dict:
    """Include constant blur / activation implementation semantics in identity."""
    shape = {key: deepcopy(node[key]) for key in (*SHAPE_KEYS, *SEMANTIC_KEYS) if key in node}
    if shape.get("op") == "conv":
        shape.setdefault("dilation", 1)
        shape.setdefault("expanded_channels", 0)
        shape.setdefault("H", shape["input_shapes"][0][1])
        shape.setdefault("W", shape["input_shapes"][0][2])
        shape.setdefault("operator_type", "DW" if shape.get("groups") == shape.get("Cin") == shape.get("Cout")
                         and shape.get("groups", 1) > 1 else "GroupConv" if shape.get("groups", 1) > 1 else "Conv")
        # A folded BN supplies the deployed convolution's bias even if the
        # unfused source Conv2d originally had bias=False (the old T6 schema).
        shape["bias"] = bool(shape.get("bias") or shape.get("bn_folded"))
    if shape.get("op") in ("add", "residual_add"):
        shape.update(op="requant_add", operator_type="requant_add")
    if shape.get("op") == "requant_add":
        shape.update(activation="linear", alignment="requant_each_input_to_common_scale")
    if shape.get("op") in ("maxpool", "relu", "silu", "body8_to_head16"):
        shape.setdefault("operator_type", shape["op"])
        shape.setdefault("activation", shape["op"] if shape["op"] in ("relu", "silu") else "linear")
    if shape.get("op") == "nearest":
        if node.get("factor", 2) != 2:
            raise ValueError("Only exact nearest factor-two is registered")
        shape.setdefault("operator_type", "nearest")
        shape.setdefault("interpolation", "nearest_exact_factor_two")
    for dims in [*shape.get("input_shapes", []), shape.get("output_shape", [])]:
        if len(dims) != 3 or any(type(x) is not int or x <= 0 for x in dims):
            raise ValueError("Primitive tensors require positive CHW dimensions")
    if not shape.get("input_shapes") or not shape.get("output_shape") or not shape.get("op"):
        raise ValueError("Incomplete primitive shape")
    return shape


def precision_for(node: Mapping, profile: str) -> dict:
    if profile not in PROFILES:
        raise ValueError("Unknown precision profile")
    if "precision" in node:
        value = deepcopy(node["precision"])
    elif "input_bits" in node:
        value = dict(input_bitwidth=node["input_bits"], output_bitwidth=node["output_bits"],
                     weight_bitwidth=node.get("weight_bits"), accumulator_bitwidth=node.get("accumulator_bits", 32))
    else:
        head = str(node.get("id", "")).startswith("head.")
        bits = 16 if head and profile != "W8A8" else 8
        weight = 16 if head and profile == "HEAD_W16A16" else 8
        value = dict(input_bitwidth=bits, output_bitwidth=bits,
                     weight_bitwidth=weight if node["op"] in ("conv", "blur") else None,
                     accumulator_bitwidth=32)
        if node["op"] == "body8_to_head16":
            value.update(input_bitwidth=8, output_bitwidth=16, weight_bitwidth=None)
    if set(value) != {"input_bitwidth", "output_bitwidth", "weight_bitwidth", "accumulator_bitwidth"}:
        raise ValueError("Precision schema mismatch")
    if any(value[k] not in (8, 16) for k in ("input_bitwidth", "output_bitwidth")) or value["accumulator_bitwidth"] != 32:
        raise ValueError("Unsupported activation/accumulator precision")
    if value["weight_bitwidth"] not in (None, 8, 16):
        raise ValueError("Unsupported weight precision")
    return value


def primitive_record(node: Mapping, profile: str) -> dict:
    shape, precision = primitive_shape(node), precision_for(node, profile)
    identity = dict(shape=shape, precision=precision)
    # Keep the historical key literally; do not pretend schema normalization
    # was already present in the old archive.
    legacy = {key: deepcopy(node[key]) for key in SHAPE_KEYS if key in node}
    if legacy.get("op") in ("add", "residual_add"):
        legacy.update(op="requant_add", operator_type="requant_add")
    return dict(primitive_id=digest(identity), **identity,
                legacy_primitive_id=digest(dict(shape=legacy, precision=precision)),
                legacy_reuse_safe=not any(key in shape for key in ("fixed_kernel", "lut_sha256", "channel_order")),
                numeric_contract_sha256=node.get("numeric_contract_sha256", node.get("numeric_contract_hash")),
                interface_contract_sha256=node.get("interface_contract_sha256", node.get("interface_contract_hash")))


def graph_requirements(graph: Mapping, profile: str) -> list[dict]:
    if profile not in PROFILES:
        raise ValueError("Unknown profile")
    declared = graph.get("profile", graph.get("precision_profile"))
    if declared is not None and declared != profile:
        raise ValueError("Graph was exported for another precision profile")
    nodes = deepcopy(graph["nodes"])
    # A2 is explicitly structural.  Real numerical graphs must export their own
    # boundaries; inventing a precision conversion for them would alter semantics.
    static = str(graph.get("graph_scope", "")).startswith("STATIC_")
    if static and profile != "W8A8":
        boundaries = {}
        for level, source in graph["neck_outputs"].items():
            name = f"head.input.{level}.input_requant"
            boundaries[source] = name
            nodes.append(dict(id=name, op="body8_to_head16", operator_type="body8_to_head16",
                              inputs=[source], input_shapes=[graph["edges"][source]],
                              output_shape=graph["edges"][source], activation="linear"))
        for node in nodes:
            if str(node["id"]).startswith("head.") and node["op"] != "body8_to_head16":
                node["inputs"] = [boundaries.get(source, source) for source in node.get("inputs", [])]
    elif not static and profile != "W8A8" and declared is None:
        raise ValueError("Mixed precision requires an explicitly exported numerical graph")
    names = [node["id"] for node in nodes]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate graph node")
    records = []
    for node in nodes:
        record = primitive_record(node, profile)
        record.update(node_id=node["id"], inputs=node.get("inputs", []))
        # A graph-level numerical contract is inherited only when actually
        # supplied by its exporter, never from a profile name.
        record["numeric_contract_sha256"] = record["numeric_contract_sha256"] or graph.get("numeric_contract_sha256", graph.get("numeric_contract_hash"))
        record["interface_contract_sha256"] = record["interface_contract_sha256"] or graph.get("interface_contract_sha256")
        records.append(record)
    return records


def build_inventory(identity: Mapping, graphs: Mapping, profiles=PROFILES) -> dict:
    """Export structural demand; no model search/evaluation or cost admission."""
    if identity.get("space_version") != SPACE:
        raise ValueError("192-v2 identity required")
    if "nodes" in graphs:
        graphs = {identity.get("architecture_id", identity.get("structure_id", "single")): graphs}
    primitives, topology = {}, []
    for architecture_id, graph in graphs.items():
        if graph.get("space_version", SPACE) != SPACE:
            raise ValueError("Old-space graph cannot enter the 192 inventory")
        for profile in profiles:
            requirements = graph_requirements(graph, profile)
            for row in requirements:
                key = row["primitive_id"]
                if key not in primitives:
                    primitives[key] = {k: deepcopy(v) for k, v in row.items() if k not in ("node_id", "inputs")}
                    primitives[key].update(profiles=[], architecture_ids=[], source_nodes=[])
                target = primitives[key]
                for field, value in (("profiles", profile), ("architecture_ids", architecture_id), ("source_nodes", row["node_id"])):
                    if value not in target[field]:
                        target[field].append(value)
            topology.append(dict(architecture_id=architecture_id, profile=profile,
                                 full_contract_sha256=graph.get("full_contract_sha256"),
                                 graph_sha256=digest(graph),
                                 numerical_contract_bound=all(row["numeric_contract_sha256"] and row["interface_contract_sha256"] for row in requirements),
                                 nodes=[dict(node_id=row["node_id"], primitive_id=row["primitive_id"]) for row in requirements]))
    result = dict(schema_version="sonar192_hardware_inventory_v1", space_version=SPACE,
                  identity=deepcopy(identity), identity_binding_sha256=digest(identity),
                  profiles=list(profiles), architecture_count=len(graphs),
                  primitive_count=len(primitives), primitives=sorted(primitives.values(), key=lambda row: row["primitive_id"]),
                  graphs=topology, scope="STRUCTURAL_DEMAND_NOT_MEASUREMENT_OR_SEARCH")
    result["inventory_sha256"] = digest(result)
    return result


def validate_mapping(mapping: Mapping) -> dict:
    keys = {"PE", "SIMD", "buffer_profile", "implementation_id"}
    options = {"tile_h", "tile_w", "numeric_contract_sha256", "interface_contract_sha256", "status"}
    if not isinstance(mapping, Mapping) or not keys <= set(mapping) or set(mapping) - keys - options:
        raise ValueError("Concrete mapping requires PE/SIMD/buffer_profile/implementation_id and registered options")
    if ("tile_h" in mapping) != ("tile_w" in mapping):
        raise ValueError("Both tile dimensions must be explicit together")
    numeric_keys = ("PE", "SIMD") + (("tile_h", "tile_w") if "tile_h" in mapping else ())
    if any(type(mapping[k]) is not int or mapping[k] <= 0 for k in numeric_keys):
        raise ValueError("Invalid concrete mapping dimensions")
    if any(not isinstance(mapping[k], str) or not mapping[k] for k in ("buffer_profile", "implementation_id")):
        raise ValueError("Mapping implementation identity is absent")
    # Candidate identities also carry document hashes/status.  They are checked
    # separately and are not a different physical implementation of the tile.
    return {key: deepcopy(mapping[key]) for key in (*sorted(keys), "tile_h", "tile_w") if key in mapping}


def formal_rejection_reasons(record: Mapping) -> list[str]:
    reasons = []
    for key in ("formal_cost_eligible", "hardware_measurement_eligible", "protocol_conditions_met"):
        if record.get(key) is not True:
            reasons.append(key.upper() + "_FALSE")
    if record.get("purpose") != "formal":
        reasons.append("DIAGNOSTIC_OR_UNDECLARED_PURPOSE")
    if record.get("protocol_status") != "FROZEN":
        reasons.append("PROTOCOL_NOT_FROZEN")
    if record.get("scope") != "BOARD_OPERATOR_MEASUREMENT":
        reasons.append("NOT_A_FORMAL_OPERATOR_MEASUREMENT")
    if not record.get("numeric_contract_sha256") or not record.get("interface_contract_sha256"):
        reasons.append("NUMERIC_OR_INTERFACE_CONTRACT_MISSING")
    return reasons


def numeric_precision_matches(numeric: Mapping, primitive: Mapping) -> bool:
    """Resolve actual mixed graph regions without relabeling the global contract."""
    precision = primitive["precision"]
    if "activation_bits" in numeric:
        return (numeric["activation_bits"] == precision["input_bitwidth"] == precision["output_bitwidth"]
                and numeric.get("accumulator_bits") == precision["accumulator_bitwidth"] == 32
                and (precision["weight_bitwidth"] is None or numeric.get("weight_bits") == precision["weight_bitwidth"]))
    if "body" in numeric and "head_activation_bits" in numeric:
        if numeric.get("accumulator_bits") != precision["accumulator_bitwidth"]:
            return False
        if primitive["shape"]["op"] in ("requantize", "body8_to_head16"):
            return (precision["input_bitwidth"] == numeric.get("body_activation_bits") == 8
                    and precision["output_bitwidth"] == numeric.get("head_activation_bits") == 16
                    and precision["weight_bitwidth"] is None)
        body = (precision["input_bitwidth"] == precision["output_bitwidth"] == numeric.get("body_activation_bits")
                and (precision["weight_bitwidth"] is None or precision["weight_bitwidth"] == numeric.get("body_weight_bits")))
        head = (precision["input_bitwidth"] == numeric.get("head_input_bits") == precision["output_bitwidth"]
                == numeric.get("head_activation_bits")
                and (precision["weight_bitwidth"] is None or precision["weight_bitwidth"] == numeric.get("head_weight_bits")))
        return body or head
    return numeric.get("precision") == precision


class VerifiedRecords(list):
    """Only source importers create this list; ordinary JSON rows are rejected."""
    def __init__(self, rows, *, token=None):
        if token is not _VERIFIED:
            raise ValueError("Use a source-verifying measurement importer")
        super().__init__(deepcopy(rows))
        self.content_sha256 = digest(list(self))

    def verify_unchanged(self):
        if digest(list(self)) != self.content_sha256:
            raise ValueError("Imported measurement rows were modified")
        for row in self:
            for binding in row["source_bindings"].values():
                checked_file(binding)


def import_t6_attempt(attempt: str | Path, *, replay: bool = True) -> VerifiedRecords:
    """Read and replay every selected T6 run; preserve all qualification flags.

    replay=False permits inspection only and deliberately cannot produce a
    VerifiedRecords object.  Numerical consumption always replays raw evidence.
    """
    if replay is not True:
        raise ValueError("Numerical cost consumption requires raw evidence replay")
    from tools.hardware_template_coverage_v1.timing_validation import (
        validate_timing_snapshot, compare_imported_timing_audit)
    from tools.hardware_template_coverage_v1.board_runner import _route_identity
    from tools.hardware_template_coverage_v1.coverage_report import route_resources
    attempt = Path(attempt).resolve(strict=True)
    rows = []
    for pointer_path in sorted((attempt / "cases").glob("*/board/selected_run.json")):
        folder = pointer_path.parent.parent
        case = read_json(folder / "case.json")
        pointer = read_json(pointer_path)
        if pointer.get("evidence_kind") != "BOARD_MEASUREMENT":
            raise ValueError("Selected T6 run is not board evidence")
        output = (pointer_path.parent / pointer["run_directory"]).resolve(strict=True)
        if not output.is_relative_to(pointer_path.parent.resolve()):
            raise ValueError("Selected run escapes case directory")
        imported = read_json(output / "verified_snapshot_import.json")
        summary = read_json(output / "summary.json")
        snapshot = Path(imported["snapshot"]).resolve(strict=True)
        if not snapshot.is_relative_to(attempt / "board_snapshots"):
            raise ValueError("T6 snapshot escapes its attempt")
        candidates = []
        for path in (attempt / "board_packages" / case["case_id"]).glob("*/case/board_bundle.json"):
            if sha256_file(path) == summary["board_bundle_sha256"]:
                candidates.append(path.parent)
        if len(candidates) != 1:
            raise ValueError("Expected exactly one source-bound T6 board package")
        package = candidates[0]
        _route_identity(package)
        checked = validate_timing_snapshot(snapshot, package)
        compare_imported_timing_audit(imported, checked)
        if imported["summary_sha256"] != sha256_file(output / "summary.json") or imported["raw_csv_sha256"] != sha256_file(output / "timing_raw.csv"):
            raise ValueError("Copied T6 summary/raw differs from its verified import")
        if checked["numerical_status"] != "PASS":
            raise ValueError("Selected T6 numerical gates failed")
        case_summary = next(row for row in summary["cases"] if row["case_id"] == case["case_id"])
        timing = read_json(output / "timing_manifest.json")
        mapping = dict(PE=case["mapping"]["PE"], SIMD=case["mapping"]["SIMD"],
                       buffer_profile=case["mapping"]["buffer_profile"],
                       implementation_id="t6_kernel_sha256:" + sha256_file(package / "kernel.cpp"),
                       tile_h=case["mapping"]["TILE_H"], tile_w=case["mapping"]["TILE_W"])
        precision = dict(input_bitwidth=case["input_bits"], output_bitwidth=case["output_bits"],
                         weight_bitwidth=case["weight_bits"], accumulator_bitwidth=case["accumulator_bits"])
        node = dict(case["shape"], id=case["source_id"], precision=precision)
        primitive = primitive_record(node, case["profile"])
        util = package / "evidence/route/utilization.rpt"
        resources = route_resources(util)
        paths = dict(pointer=pointer_path, case=folder / "case.json", summary=output / "summary.json",
                     raw=output / "timing_raw.csv", import_audit=output / "verified_snapshot_import.json",
                     numeric=package / "numeric_contract.json", kernel=package / "kernel.cpp",
                     bundle=package / "board_bundle.json", timing_manifest=output / "timing_manifest.json",
                     snapshot_manifest=snapshot / "evidence/snapshot_manifest.json",
                     bitstream=package / "vivado_shell/top.bit")
        if util.is_file():
            paths["route_utilization"] = util
        row = dict(primitive, record_id=case["case_id"], profile=case["profile"], mapping=mapping,
                   platform=case["platform"], nominal_clock_hz=timing["cases"][0]["clock_hz"],
                   actual_clock_hz=[item["clock_hz"] for item in timing["batches"]],
                   numeric_contract_sha256=digest(read_json(package / "numeric_contract.json")),
                   source_numeric_contract_file_sha256=case["numeric_contract_sha256"],
                   interface_contract_sha256=None,
                   runtime_protocol=checked.get("runtime_protocol", {}).get("runtime_protocol_id", "LEGACY_UNSPECIFIED"),
                   metrics={**{name + "_us": case_summary["scopes"][scope]["mean_ns"] / 1000
                               for name, scope in (("compute", "compute"), ("transfer", "transfer"), ("host", "call"))},
                            **{name: resources.get(name) for name in ("LUT", "DSP", "BRAM18")}},
                   measured_calls=case_summary["measured_rows"], independent_batches=case_summary["independent_batches"],
                   protocol_status=summary["protocol_status"],
                   formal_cost_eligible=summary["formal_cost_eligible"],
                   hardware_measurement_eligible=summary["hardware_measurement_eligible"],
                   protocol_conditions_met=summary["protocol_conditions_met"],
                   purpose="diagnostic", scope="TEMPLATE_COVERAGE_VALIDATION_NOT_FORMAL_COST_TABLE",
                   condition_deviations=summary.get("condition_deviations", []),
                   raw_replay_status=checked["status"], source_bindings={name: bind(path) for name, path in paths.items()},
                   resource_scope="standalone_operator_with_shell_not_integrated_network",
                   latency_scope="separate_PL_compute_PL_transfer_host_call")
        row["formal_rejection_reasons"] = formal_rejection_reasons(row)
        rows.append(row)
    return VerifiedRecords(rows, token=_VERIFIED)


def import_board_receipts(paths, *, formal: bool = True) -> VerifiedRecords:
    """Consume future v1 receipts with raw samples and independently bound review.

    Receipt production is deliberately outside this module: importing does not
    execute a board, rewrite protocols, or approve exceptions. Raw metrics are
    recomputed; the receipt cannot substitute a fabricated derived cost value.
    """
    rows = []
    for path in map(Path, paths):
        receipt = read_json(path)
        if receipt.get("schema_version") != "sonar192_board_operator_receipt_v1":
            raise ValueError("Unsupported board receipt schema")
        evidence = receipt["evidence"]
        required = {"raw_csv", "protocol", "mapping_plan", "numeric_contract", "interface_contract",
                    "route_resources", "bitstream", "hwh", "source_manifest", "review", "correctness"}
        if set(evidence) != required:
            raise ValueError("Incomplete or unregistered receipt evidence")
        files = {name: checked_file(value, path.parent) for name, value in evidence.items()}
        protocol, plan = read_json(files["protocol"]), read_json(files["mapping_plan"])
        if protocol.get("purpose") != receipt.get("purpose"):
            raise ValueError("Receipt purpose differs from the independently bound protocol")
        if formal and (protocol.get("scope") != "BOARD_OPERATOR_MEASUREMENT"
                       or protocol.get("minimum_batches", 0) < 3
                       or protocol.get("minimum_warmup", 0) < 50
                       or protocol.get("minimum_measured_per_batch", 0) < 1000
                       or not 0 < protocol.get("maximum_pl_cv", 1) <= .05):
            raise ValueError("Formal measurement protocol is incomplete or weakens required gates")
        mapping = validate_mapping(receipt["mapping"])
        if validate_mapping(plan.get("mapping")) != mapping or plan.get("status") != "FROZEN_CONCRETE_MAPPING":
            raise ValueError("Concrete mapping has not been frozen exactly")
        primitive = primitive_record(dict(receipt["shape"], id="operator", precision=receipt["precision"]), receipt["profile"])
        if receipt.get("primitive_id") != primitive["primitive_id"]:
            raise ValueError("Receipt primitive identity mismatch")
        review = read_json(files["review"])
        reviewed = {key: value for key, value in evidence.items() if key != "review"}
        measurement_key = dict(primitive_id=primitive["primitive_id"], mapping=mapping,
                               platform=receipt["platform"], profile=receipt["profile"])
        if (review.get("status") != "PASS_SOURCE_BOUND_BOARD_MEASUREMENT"
                or review.get("evidence_sha256") != digest(reviewed)
                or review.get("measurement_key_sha256") != digest(measurement_key)
                or review.get("purpose") != receipt.get("purpose")):
            raise ValueError("Independent board review is not bound to these exact sources/results")
        source_manifest = read_json(files["source_manifest"])
        if not source_manifest.get("files") or source_manifest.get("purpose") != receipt.get("purpose"):
            raise ValueError("Empty implementation source manifest")
        for binding in source_manifest["files"].values():
            checked_file(binding, files["source_manifest"].parent)
        if any(files[key].stat().st_size == 0 for key in ("bitstream", "hwh")):
            raise ValueError("Empty bitstream/HWH")
        numeric, interface = read_json(files["numeric_contract"]), read_json(files["interface_contract"])
        if (not numeric_precision_matches(numeric, primitive) or interface.get("precision") != primitive["precision"]
                or numeric.get("profile", receipt["profile"]) != receipt["profile"]
                or interface.get("numeric_contract_sha256") != digest(numeric)):
            raise ValueError("Numeric/interface precision or document binding mismatch")
        resources = read_json(files["route_resources"])
        if (resources.get("stage") != "POST_ROUTE" or resources.get("part") != receipt["platform"]["part"]
                or resources.get("bitstream_sha256") != sha256_file(files["bitstream"])
                or resources.get("hwh_sha256") != sha256_file(files["hwh"])
                or not _finite(resources.get("WNS_ns"))):
            raise ValueError("Route/bitstream/HWH identity or timing mismatch")
        with files["raw_csv"].open(encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
        if not raw:
            raise ValueError("Empty raw measurement CSV")
        batches = defaultdict(list)
        clocks = []
        for sample in raw:
            if sample.get("valid") != "true" or sample.get("fixture_match") != "true":
                raise ValueError("Invalid / numerically incorrect raw sample cannot be dropped")
            if sample.get("phase") not in ("warmup", "measure") or not sample.get("batch_id"):
                raise ValueError("Raw sample phase/batch missing")
            values = [float(sample[key]) for key in ("compute_ns", "transfer_ns", "host_ns", "clock_hz")]
            if any(not _finite(value, positive=True) for value in values) or values[2] < values[0] + values[1]:
                raise ValueError("Raw timings/clock are invalid or host does not contain PL intervals")
            clocks.append(values[3])
            if formal:
                allowed_cpu = protocol.get("allowed_cpu_governors", [])
                if not allowed_cpu or sample.get("cpu_governor") not in allowed_cpu:
                    raise ValueError("CPU condition is outside the frozen protocol")
                if abs(values[3] / receipt["nominal_clock_hz"] - 1) > protocol.get("maximum_clock_relative_deviation", 0):
                    raise ValueError("Observed clock violates the frozen protocol")
            batches[sample["batch_id"]].append(sample)
        if len(batches) < protocol["minimum_batches"]:
            raise ValueError("Insufficient independent batches")
        correctness = read_json(files["correctness"])
        if correctness.get("purpose") != receipt["purpose"] or not correctness.get("fixtures"):
            raise ValueError("Correctness fixtures are missing or have a different evidence purpose")
        fixture_batches, correctness_sources = defaultdict(set), {}
        for index, fixture in enumerate(correctness["fixtures"]):
            if fixture.get("batch_id") not in batches or not fixture.get("fixture_id"):
                raise ValueError("Correctness fixture batch/identity mismatch")
            if fixture["fixture_id"] in fixture_batches[fixture["batch_id"]]:
                raise ValueError("Duplicate correctness fixture")
            fixture_batches[fixture["batch_id"]].add(fixture["fixture_id"])
            expected = checked_file(fixture["expected"], files["correctness"].parent)
            actual = checked_file(fixture["actual"], files["correctness"].parent)
            expected_bytes = math.prod(primitive["shape"]["output_shape"]) * primitive["precision"]["output_bitwidth"] // 8
            if expected.stat().st_size != expected_bytes or expected.read_bytes() != actual.read_bytes():
                raise ValueError("Full primitive output is not bit-exact or has wrong byte count")
            correctness_sources[f"fixture_{index}_expected"] = bind(expected)
            correctness_sources[f"fixture_{index}_actual"] = bind(actual)
        minimum_fixtures = protocol.get("minimum_correctness_fixtures", 5 if formal else 1)
        if formal and minimum_fixtures < 5:
            raise ValueError("Formal correctness requires all five full-output fixtures")
        if set(fixture_batches) != set(batches) or any(len(items) < minimum_fixtures for items in fixture_batches.values()):
            raise ValueError("Incomplete correctness coverage across independent batches")
        measured = []
        for batch, samples in batches.items():
            warm = [s for s in samples if s["phase"] == "warmup"]
            keep = [s for s in samples if s["phase"] == "measure"]
            if len(warm) < protocol["minimum_warmup"] or len(keep) < protocol["minimum_measured_per_batch"]:
                raise ValueError("Insufficient warmup or formal samples in batch " + batch)
            measured.extend(keep)
        if formal:
            independence = [{sample.get("independence_id") for sample in samples} for samples in batches.values()]
            if any(len(values) != 1 or not next(iter(values)) for values in independence) or len(set.union(*independence)) != len(batches):
                raise ValueError("Batches are not bound to distinct runtime reload/restart events")
        metrics = {name: statistics.mean(float(row[column]) for row in measured) / 1000
                   for name, column in (("compute_us", "compute_ns"), ("transfer_us", "transfer_ns"), ("host_us", "host_ns"))}
        for name in ("LUT", "DSP", "BRAM18"):
            if not _finite(resources.get(name)):
                raise ValueError("Missing/nonfinite post-route resource " + name)
            metrics[name] = resources[name]
        for key in ("compute_ns", "transfer_ns"):
            values = [float(row[key]) for row in measured]
            if len(values) < 2 or statistics.stdev(values) / statistics.mean(values) > protocol["maximum_pl_cv"]:
                raise ValueError("PL timing variation exceeds the bound protocol")
        row = dict(primitive, record_id=receipt["record_id"], profile=receipt["profile"], mapping=mapping,
                   platform=receipt["platform"], nominal_clock_hz=receipt["nominal_clock_hz"], actual_clock_hz=clocks,
                   numeric_contract_sha256=digest(numeric),
                   interface_contract_sha256=digest(interface),
                   runtime_protocol=protocol["runtime_protocol"], metrics=metrics,
                   measured_calls=len(measured), independent_batches=len(batches),
                   protocol_status=protocol["status"], purpose=receipt["purpose"], scope=receipt["scope"],
                   formal_cost_eligible=receipt["formal_cost_eligible"],
                   hardware_measurement_eligible=receipt["hardware_measurement_eligible"],
                   protocol_conditions_met=receipt["protocol_conditions_met"],
                   source_bindings={"receipt": bind(path), **{name: bind(value) for name, value in files.items()},
                                    **correctness_sources,
                                    **{"implementation_" + name: bind(checked_file(value, files["source_manifest"].parent))
                                       for name, value in source_manifest["files"].items()}},
                   raw_replay_status="PASS_RAW_CSV_RECOMPUTATION",
                   resource_scope="standalone_operator_with_shell_not_integrated_network",
                   latency_scope="separate_PL_compute_PL_transfer_host_call")
        row["formal_rejection_reasons"] = formal_rejection_reasons(row)
        if formal and row["formal_rejection_reasons"]:
            raise ValueError("Formal cost admission denied: " + ",".join(row["formal_rejection_reasons"]))
        if formal:
            row["formal_receipt_registration"] = require_registered_board_receipt(path, files["review"])
        rows.append(row)
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate measurement record")
    return VerifiedRecords(rows, token=_VERIFIED)


def _record_key(row):
    return (row["primitive_id"], digest(row["mapping"]), row.get("numeric_contract_sha256"),
            row.get("interface_contract_sha256"))


class BoardCostBackend:
    def __init__(self, identity: Mapping, inventory: Mapping | None, records: VerifiedRecords, *, formal=True):
        if not isinstance(records, VerifiedRecords):
            raise ValueError("Cost backend requires source-verified records, not JSON flags")
        records.verify_unchanged()
        if identity.get("space_version") != SPACE:
            raise ValueError("Hardware backend only supports the frozen 192-v2 space")
        if formal:
            from .contract import validate_identity
            validate_identity(identity)
            if identity.get("purpose") != "formal":
                raise ValueError("Formal backend requires a formal identity")
        self.identity, self.inventory, self.records, self.formal = deepcopy(identity), deepcopy(inventory), records, formal

    def query(self, graph: Mapping, profile: str, mapping: Mapping) -> dict:
        self.records.verify_unchanged()
        mapping = validate_mapping(mapping)
        if graph.get("space_version", SPACE) != SPACE:
            raise ValueError("Wrong-space graph")
        if self.formal and graph.get("full_contract_sha256") != self.identity.get("full_contract_sha256"):
            raise ValueError("Graph/model contract mismatch")
        if self.formal and (profile != self.identity["profile"] or mapping != validate_mapping(self.identity["mapping"])):
            raise ValueError("Cost query profile/mapping differs from the candidate identity")
        requirements = graph_requirements(graph, profile)
        selected, missing, reasons = [], [], []
        for req in requirements:
            pool = [row for row in self.records if row["primitive_id"] == req["primitive_id"] and row["mapping"] == mapping]
            if self.formal:
                pool = [row for row in pool if not formal_rejection_reasons(row)
                        and row["numeric_contract_sha256"] == req["numeric_contract_sha256"]
                        and row["interface_contract_sha256"] == req["interface_contract_sha256"]]
            if len(pool) > 1:
                raise ValueError("Ambiguous repeated measurement: freeze a selection before query")
            if not pool:
                missing.append(dict(node_id=req["node_id"], primitive_id=req["primitive_id"],
                                    reason="NO_EXACT_ELIGIBLE_PRIMITIVE_MAPPING_CONTRACT"))
            else:
                selected.append(pool[0])
        if self.formal and any(not row["numeric_contract_sha256"] or not row["interface_contract_sha256"] for row in requirements):
            reasons.append("GRAPH_NUMERIC_INTERFACE_CONTRACT_NOT_BOUND")
        if missing:
            reasons.append("MISSING_MEASUREMENTS")
        cost = None
        if not reasons:
            cost = {name: sum(row["metrics"][name] for row in selected) for name in METRICS[:3]}
            for name in METRICS[3:]:
                cost[name] = None if any(row["metrics"].get(name) is None for row in selected) else max(row["metrics"][name] for row in selected)
        return dict(status="BLOCKED" if reasons else "LOOKED_UP" if self.formal else "DIAGNOSTIC_LOOKUP",
                    formal_eligible=self.formal and not reasons, identity_sha256=self.identity.get("identity_sha256"),
                    graph_sha256=digest(graph), profile=profile, mapping=mapping, cost=cost,
                    reasons=reasons, missing=missing, graph_nodes=len(requirements), matched_nodes=len(selected),
                    record_ids=[row["record_id"] for row in selected],
                    latency_scope="serial_operator_sum_no_overlap_not_network_e2e",
                    resource_scope="peak_standalone_shell_not_integrated_network",
                    diagnostic_numeric_contract_match_required=self.formal)


def query_graph_cost(graph, records, profile, mapping, *, formal=True, identity=None):
    return BoardCostBackend(identity, None, records, formal=formal).query(graph, profile, mapping)


def coverage_queue(inventory: Mapping, records: VerifiedRecords, mappings: Mapping, *, active_profiles=PROFILES[:2]) -> dict:
    """Pure demand generation; an absent concrete policy becomes an explicit gap."""
    records.verify_unchanged()
    if not set(active_profiles) <= set(PROFILES) or not active_profiles:
        raise ValueError("Explicit registered active measurement profiles required")
    tasks, admitted = {}, 0
    for primitive in inventory["primitives"]:
        for policy in POLICIES:
            mapping = mappings.get(policy)
            if mapping is not None:
                mapping = validate_mapping(mapping)
            task_key = dict(primitive_id=primitive["primitive_id"], mapping=mapping)
            if mapping is None:
                task_key["unfrozen_policy"] = policy
            task_id = digest(task_key)
            if task_id in tasks:
                tasks[task_id]["policies"].append(policy)
                continue
            matching = [row for row in records if row["primitive_id"] == primitive["primitive_id"] and row["mapping"] == mapping]
            eligible = [row for row in matching if not formal_rejection_reasons(row)
                        and row["numeric_contract_sha256"] == primitive["numeric_contract_sha256"]
                        and row["interface_contract_sha256"] == primitive["interface_contract_sha256"]]
            admitted += bool(eligible)
            tasks[task_id] = dict(task_id=task_id, **task_key, policies=[policy],
                                 status="AVAILABLE" if eligible else "MAPPING_NOT_FROZEN" if mapping is None else "NEEDS_MEASUREMENT",
                                 shape=primitive["shape"], precision=primitive["precision"],
                                 profiles=primitive["profiles"],
                                 conditional_only=not bool(set(primitive["profiles"]) & set(active_profiles)),
                                 numerical_contract_bound=bool(primitive["numeric_contract_sha256"] and primitive["interface_contract_sha256"]),
                                 observed_records=[row["record_id"] for row in matching],
                                 rejected_observations={row["record_id"]: formal_rejection_reasons(row) for row in matching})
    return dict(schema_version="sonar192_measurement_queue_v1", inventory_sha256=inventory["inventory_sha256"],
                tasks=list(tasks.values()), task_count=len(tasks), eligible_task_count=admitted,
                active_profiles=list(active_profiles), active_task_count=sum(not row["conditional_only"] for row in tasks.values()),
                conditional_task_count=sum(row["conditional_only"] for row in tasks.values()),
                formal_search_ready=bool(tasks) and admitted == len(tasks),
                execution="NOT_STARTED_NO_BOARD_HLS_OR_SEARCH_ACTION")


def _prediction_family(row: Mapping) -> tuple:
    shape = row["shape"]
    # Kernel/stride/groups relative to channels and activation remain in-family;
    # shape is held out, while numeric/platform/mapping/protocol cannot drift.
    group_kind = "depthwise" if shape.get("groups") == shape.get("Cin") == shape.get("Cout") and shape.get("groups", 1) > 1 else shape.get("groups", 1)
    family = {key: shape.get(key) for key in ("op", "kernel", "stride", "activation", "dilation", "fixed_kernel", "denominator", "fixed_kernel_denominator", "channel_order", "lut_sha256")}
    family["group_kind"] = group_kind
    return (digest(family), digest(row["precision"]), digest(row["mapping"]), digest(row["platform"]),
            row["nominal_clock_hz"], row["numeric_contract_sha256"], row["interface_contract_sha256"], row["runtime_protocol"])


def _features(row):
    shape = row["shape"]
    return [math.log2(value) for dims in [*shape["input_shapes"], shape["output_shape"]] for value in dims]


class NearestMeasuredPredictor:
    """Conservative local baseline: same implementation/contract, nearest shape.

    No unmeasured family fallback and no network performance claim.  This is a
    real estimator consuming measurements, not a marker named 'predictor'.
    """
    def __init__(self, records: VerifiedRecords, *, formal=False):
        if not isinstance(records, VerifiedRecords):
            raise ValueError("Predictor requires verified source records")
        records.verify_unchanged()
        self.records, self.formal = records, formal
        if formal and any(formal_rejection_reasons(row) for row in records):
            raise ValueError("Ineligible observations cannot train a formal predictor")

    def predict(self, query: Mapping, *, excluded_primitive_ids=()) -> dict:
        pool = [row for row in self.records if _prediction_family(row) == _prediction_family(query)
                and row["primitive_id"] not in excluded_primitive_ids]
        if not pool:
            return dict(status="UNSUPPORTED_FAMILY", prediction=None, source_record_ids=[])
        features = _features(query)
        pool = [row for row in pool if len(_features(row)) == len(features)]
        if not pool:
            return dict(status="UNSUPPORTED_LAYOUT", prediction=None, source_record_ids=[])
        row = min(pool, key=lambda candidate: (sum((a-b)**2 for a,b in zip(features, _features(candidate))), candidate["record_id"]))
        return dict(status="PREDICTED", prediction=deepcopy(row["metrics"]), source_record_ids=[row["record_id"]],
                    method="nearest_log2_shape_within_exact_family_contract_mapping_platform_protocol",
                    formal_eligible=False, validation_required=True)

    def manifest(self) -> dict:
        result = dict(schema_version="sonar192_nearest_measured_predictor_v1", method="nearest_log2_shape",
                      source_records_sha256=self.records.content_sha256,
                      record_ids=[row["record_id"] for row in self.records],
                      source_bindings={row["record_id"]: row["source_bindings"] for row in self.records},
                      family_rule="exact_operator_kernel_stride_activation_numeric_interface_mapping_platform_clock_runtime_protocol",
                      fallback="ABSTAIN_NO_CROSS_FAMILY_ESTIMATES", formal_search_ready=False,
                      purpose="formal_validation_pending" if self.formal else "diagnostic")
        result["model_sha256"] = digest(result)
        return result


def validate_predictor(records: VerifiedRecords, *, formal=False) -> dict:
    """Leave one complete primitive shape out, never fit on its repeated runs."""
    predictor = NearestMeasuredPredictor(records, formal=formal)
    predictions = []
    for row in records:
        pred = predictor.predict(row, excluded_primitive_ids={row["primitive_id"]})
        errors = {}
        if pred["prediction"] is not None:
            for metric in METRICS:
                actual, estimated = row["metrics"].get(metric), pred["prediction"].get(metric)
                if actual is None or estimated is None:
                    continue
                if not _finite(actual) or not _finite(estimated):
                    raise ValueError("Invalid cost metric")
                errors[metric] = dict(actual=actual, prediction=estimated, absolute_error=abs(estimated-actual),
                                      percentage_error=None if actual == 0 else 100*abs(estimated-actual)/actual)
        predictions.append(dict(record_id=row["record_id"], held_out_primitive_id=row["primitive_id"], **pred, errors=errors))
    metrics = {}
    for metric in METRICS:
        values = [row["errors"][metric] for row in predictions if metric in row["errors"]]
        percentages = [row["percentage_error"] for row in values if row["percentage_error"] is not None]
        metrics[metric] = dict(predicted_count=len(values), total_count=len(records),
                               coverage=len(values)/len(records) if records else 0,
                               MAE=statistics.mean(row["absolute_error"] for row in values) if values else None,
                               MAPE=statistics.mean(percentages) if percentages else None,
                               max_percentage_error=max(percentages) if percentages else None,
                               zero_actual_excluded_from_MAPE=len(values)-len(percentages))
    return dict(schema_version="sonar192_predictor_validation_v1", status="DIAGNOSTIC_VALIDATION" if not formal else "VALIDATION_REQUIRES_ACCEPTANCE_THRESHOLDS",
                split_unit="primitive_shape_all_repeated_measurements_held_out_together",
                method="nearest_log2_shape", metrics=metrics, predictions=predictions,
                source_records_sha256=records.content_sha256,
                formal_search_ready=False, limitation="No acceptance gate or cross-family extrapolation; standalone operators only")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t6-attempt", type=Path, default=ROOT / "results/hardware_template_coverage_20261013/attempt01")
    parser.add_argument("--graphs", type=Path, default=ROOT / "results/space192_v2_20261013/hw_graphs.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from .contract import model_contract
    identity = model_contract("9ee366")
    inventory = build_inventory(identity, read_json(args.graphs))
    inventory["source_graph_binding"] = bind(args.graphs)
    inventory["source_graph_scope"] = "A2_STATIC_SHAPES_NOT_INTEGER_EXPORT"
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = digest(inventory)
    records = import_t6_attempt(args.t6_attempt)
    report = dict(status="LOCAL_RAW_REPLAY_COMPLETE", observed_records=len(records),
                  measured_calls=sum(row["measured_calls"] for row in records),
                  independent_batches=sum(row["independent_batches"] for row in records),
                  formal_eligible_records=sum(not formal_rejection_reasons(row) for row in records),
                  rows=list(records), records_sha256=records.content_sha256,
                  source_bindings={"adapter": bind(__file__), "graphs": bind(args.graphs)},
                  execution="READ_ONLY_EXISTING_ARCHIVES_NO_NEW_MEASUREMENTS")
    # Never assign six named policies to the single T6 physical mapping without
    # a reviewed concrete plan. The queue deliberately reports those gaps.
    queue = coverage_queue(inventory, records, {})
    validation = validate_predictor(records)
    for name, value in (("inventory.json", inventory), ("board_observations.json", report),
                        ("measurement_queue.json", queue), ("predictor_validation.json", validation),
                        ("predictor_model.json", NearestMeasuredPredictor(records).manifest())):
        write_new(args.output / name, value)
    print(json.dumps(dict(output=str(args.output.resolve()), primitive_count=inventory["primitive_count"],
                          observed=len(records), formal_eligible=report["formal_eligible_records"],
                          queue_tasks=queue["task_count"], validation=validation["metrics"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
