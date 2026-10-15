"""Fail-closed, source-verified operator costs for elastic_neck72_v1.

This module imports local evidence only; it never launches HLS or estimates a
missing cost. Serial CSYNTH sums and peak kernel resources are NOT board E2E or
integrated resources. An empty manifest is valid bookkeeping, not search-ready.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

from tools.modnas_elastic.cost_graph import (ROOT, build_cost_graph, canonical_hash,
                                             primitive_shape, validate_architecture)
sys.path.insert(0, str(ROOT / "src"))
from tools.modnas_elastic.hardware import PART, _xml_metrics
from tools.modnas_elastic.model import (SEARCH_SPACE_VERSION, architecture_id,
                                      architecture_hash, enumerate_architectures)
from tools.sonar_joint_v2.policy import PROFILES, POLICIES

SCHEMA = "sonar_light_joint_cost_v1"
PLATFORM = dict(part=PART, tool_version="2022.2", target_clock_ns=5)
SOURCES = ("tools/modnas_elastic/cost_graph.py", "tools/modnas_elastic/model.py",
           "tools/modnas_elastic/hardware.py", "src/hwnas_fpga/deploy/h2_contract.py",
           "tools/sonar_joint_v2/policy.py",
           "tools/modnas_comparison/quantization.py", "tools/modnas_comparison/quantization_mixed.py",
           "tools/fpga_h4/quant_contracts.py", "tools/modnas_comparison/joint_cost_backend.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file(base, binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("File evidence requires path and sha256")
    path = Path(binding["path"])
    path = path if path.is_absolute() else base / path
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise ValueError("Missing or hash-mismatched evidence: " + str(path))
    return path.resolve()


def _bind(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


@lru_cache(maxsize=3)
def _contracts(profile):
    # Reuse the exact arithmetic implemented by the existing software adapter.
    if profile == "W8A8":
        from hwnas_fpga.deploy.h2_contract import EXPECTED_HASH, load_numeric_contract_v1
        load_numeric_contract_v1()
        body = dict(numeric_contract_hash=EXPECTED_HASH, interface_contract_hash=canonical_hash(
            _read(ROOT / "reports/fpga_h2_contract/interface_contract_v1.json")))
        closure = _read(ROOT / "reports/fpga_h2_contract/closure_manifest.json")
        if any(closure.get(key) != value for key, value in body.items()):
            raise ValueError("H2 body contract closure drift")
        return dict(body=body, head=body)
    from tools.modnas_comparison.quantization_mixed import _contracts as contracts
    return contracts(profile)


def graph_requirements(architecture, profile, num_classes=2):
    """Precision-explicit topology; includes all four mixed-head conversions."""
    if profile not in PROFILES:
        raise ValueError("Unknown precision profile")
    graph = build_cost_graph(architecture, num_classes=num_classes)
    contracts = _contracts(profile)
    nodes = []
    for source in graph["nodes"]:
        node = deepcopy(source)
        body = contracts["body"]
        node.update(precision=dict(input_bitwidth=8, weight_bitwidth=8 if node["op"] == "conv" else 0,
                                   accumulator_bitwidth=32, output_bitwidth=8),
                    numeric_contract_hash=body["numeric_contract_hash"],
                    interface_contract_hash=body["interface_contract_hash"])
        if profile != "W8A8" and node["id"].startswith("head."):
            boundary_id = node["id"].rsplit(".", 1)[0] + ".input_requant"
            boundary = dict(id=boundary_id, op="requantize", operator_type="body8_to_head16_requantize",
                inputs=node["inputs"], input_shapes=node["input_shapes"], output_shape=node["input_shapes"][0],
                activation="linear", precision=dict(input_bitwidth=8, weight_bitwidth=0,
                    accumulator_bitwidth=32, output_bitwidth=16),
                numeric_contract_hash=contracts["boundary_numeric_contract_hash"],
                # No frozen boundary transport exists. Receipt must supply a reviewed
                # interface document bound to these exact input/output precisions.
                interface_contract_hash=None)
            nodes.append(boundary)
            node.update(inputs=[boundary_id], precision=dict(input_bitwidth=16,
                weight_bitwidth=8 if profile == "HEAD_W8A16" else 16,
                accumulator_bitwidth=32, output_bitwidth=16), **contracts["head"])
        nodes.append(node)
    requirements = []
    for node in nodes:
        spec = dict(primitive=primitive_shape(node), precision=node["precision"],
                    numeric_contract_hash=node["numeric_contract_hash"],
                    interface_contract_hash=node["interface_contract_hash"], **PLATFORM)
        requirements.append(dict(node_id=node["id"], inputs=node["inputs"], requirement=spec,
                                 primitive_requirement_id=canonical_hash(spec)))
    topology = dict(architecture_hash=architecture_hash(architecture), profile=profile,
                   num_classes=num_classes, nodes=requirements, outputs=graph["outputs"],
                   input_shape=graph["input_shape"], cost_graph_sha256=graph["cost_graph_sha256"])
    return dict(topology, topology_sha256=canonical_hash(topology))


def empty_manifest(num_classes=2):
    return dict(schema_version=SCHEMA, space_version=SEARCH_SPACE_VERSION, num_classes=num_classes,
                profiles=list(PROFILES), policies=list(POLICIES), platform=PLATFORM,
                implementation_sources={name: sha(ROOT / name) for name in SOURCES},
                graph_contracts=[], mapping_bindings=[], measurements=[],
                evidence_scope="OP_LEVEL_HLS_ONLY", deployment="NOT_CLAIMED")


def _unique(rows, key, description):
    result = {}
    if not isinstance(rows, list):
        raise ValueError(description + " must be a list")
    for row in rows:
        identity = key(row)
        if identity in result:
            raise ValueError("Duplicate " + description)
        result[identity] = row
    return result


class VerifiedJointCostBackend:
    def __init__(self, path, *, num_classes, profiles, policies, space_version):
        self.path = Path(path).resolve()
        self.base = self.path.parent
        self.manifest = _read(self.path)
        self.evidence_sha256 = sha(self.path)
        expected = dict(schema_version=SCHEMA, space_version=SEARCH_SPACE_VERSION,
                        num_classes=num_classes, profiles=list(profiles), policies=list(policies),
                        platform=PLATFORM, evidence_scope="OP_LEVEL_HLS_ONLY", deployment="NOT_CLAIMED")
        if (space_version != SEARCH_SPACE_VERSION or tuple(profiles) != PROFILES or tuple(policies) != POLICIES
                or type(num_classes) is not int or num_classes < 1):
            raise ValueError("Unregistered space, profiles, policies or class count")
        for key, value in expected.items():
            if self.manifest.get(key) != value:
                raise ValueError("Joint manifest mismatch: " + key)
        if self.manifest.get("implementation_sources") != {name: sha(ROOT / name) for name in SOURCES}:
            raise ValueError("Implementation source drift; regenerate and re-review evidence manifest")
        self.num_classes = num_classes
        self.graphs = {}
        self.specs = {}
        for arch in enumerate_architectures():
            for profile in profiles:
                graph = graph_requirements(arch, profile, num_classes)
                self.graphs[architecture_hash(arch), profile] = graph
                self.specs.update({node["primitive_requirement_id"]: node["requirement"] for node in graph["nodes"]})
        self.graph_contracts = _unique(self.manifest["graph_contracts"],
            lambda row: (row["architecture_hash"], row["profile"]), "graph contract")
        for identity, row in self.graph_contracts.items():
            if identity not in self.graphs:
                raise ValueError("Unregistered graph contract")
            proof = _read(_file(self.base, row["evidence"]))
            expected_graph = self.graphs[identity]
            if proof.get("topology") != expected_graph:
                raise ValueError("Graph contract must bind complete precision topology, including mixed boundaries")
            if proof.get("status") != "REVIEWED_INTEGER_TOPOLOGY_CONTRACT":
                raise ValueError("Graph numerical contract not reviewed")
            if proof.get("implementation_sources") != self.manifest["implementation_sources"]:
                raise ValueError("Graph contract is not bound to current numerical implementation")
            self._source_files(proof.get("source_files"), proof.get("source_hash"), self.base)
        self.bindings = _unique(self.manifest["mapping_bindings"],
            lambda row: (row["primitive_requirement_id"], row["mapping_policy"]), "mapping binding")
        for (rid, policy), binding in self.bindings.items():
            if rid not in self.specs or policy not in policies:
                raise ValueError("Mapping binding has unregistered requirement/policy")
            plan = _read(_file(self.base, binding["evidence"]))
            if (plan.get("primitive_requirement_id") != rid or plan.get("mapping_policy") != policy
                    or plan.get("status") != "FROZEN_CONCRETE_MAPPING"
                    or plan.get("mapping") != binding.get("mapping")):
                raise ValueError("Mapping must have an exact, separately frozen plan")
            self._mapping(binding["mapping"])
        self.measurements = _unique(self.manifest["measurements"], lambda row: row["task_id"], "measurement")
        self.verified = {}
        tasks = self._all_tasks()
        for task_id, receipt in self.measurements.items():
            if task_id not in tasks:
                raise ValueError("Measurement does not match any frozen primitive/mapping task")
            self.verified[task_id] = self._verify_measurement(receipt, tasks[task_id])

    @staticmethod
    def _mapping(mapping):
        if not isinstance(mapping, dict) or set(mapping) != {"PE", "SIMD", "buffer_profile", "implementation_id"}:
            raise ValueError("Mapping requires PE, SIMD, buffer_profile and implementation_id")
        if any(type(mapping[name]) is not int or mapping[name] < 1 for name in ("PE", "SIMD")):
            raise ValueError("Invalid PE/SIMD")
        if any(not isinstance(mapping[name], str) or not mapping[name] for name in ("buffer_profile", "implementation_id")):
            raise ValueError("Missing mapping implementation identity")

    @staticmethod
    def _source_files(files, digest, base):
        if not isinstance(files, dict) or not files or canonical_hash(files) != digest:
            raise ValueError("Source manifest hash mismatch")
        for name, expected in files.items():
            _file(base, dict(path=name, sha256=expected))

    def _task(self, rid, policy):
        binding = self.bindings.get((rid, policy))
        mapping = binding["mapping"] if binding else None
        identity = dict(primitive_requirement_id=rid, mapping=mapping)
        # Unfrozen policies cannot alias each other. Frozen identical concrete
        # mappings may legitimately reuse the same measured primitive.
        if mapping is None:
            identity["unfrozen_mapping_policy"] = policy
        return dict(task_id=canonical_hash(identity), **identity, requirement=self.specs[rid])

    def _all_tasks(self):
        return {task["task_id"]: task for rid in self.specs for policy in POLICIES
                for task in (self._task(rid, policy),)}

    def _verify_measurement(self, receipt, task):
        if task["mapping"] is None:
            raise ValueError("Cannot import measurement before concrete mapping is frozen")
        key_path = _file(self.base, receipt["measurement_key"])
        key = _read(key_path)
        if key.get("requirement") != task["requirement"] or key.get("mapping") != task["mapping"]:
            raise ValueError("Measured primitive, precision, contract or mapping mismatch")
        if key.get("task_id") != task["task_id"] or key.get("csim_status") != "PASS_BIT_EXACT":
            raise ValueError("Measurement task binding/parity status mismatch")
        self._source_files(key.get("source_files"), key.get("source_hash"), key_path.parent)
        self._source_files(key.get("witness_files"), key.get("witness_hash"), key_path.parent)
        if not {"input.bin", "reference.bin"} <= set(key["witness_files"]):
            raise ValueError("Input and reference witness files required")
        req = task["requirement"]
        primitive, mapping = req["primitive"], task["mapping"]
        if primitive["op"] == "conv":
            if primitive["Cin"] % mapping["SIMD"] or primitive["Cout"] % mapping["PE"]:
                raise ValueError("Illegal channel divisibility in concrete mapping")
            if not {"weights.bin", "biases.bin"} <= set(key["witness_files"]):
                raise ValueError("Convolution requires parameter witnesses")
            weight_bytes = (primitive["Cout"] * (primitive["Cin"] // primitive["groups"])
                            * primitive["kernel"] ** 2 * req["precision"]["weight_bitwidth"] // 8)
            if ((key_path.parent / "weights.bin").stat().st_size != weight_bytes
                    or (key_path.parent / "biases.bin").stat().st_size != primitive["Cout"] * 4):
                raise ValueError("Convolution parameter witness byte counts mismatch")
        input_size = sum(math.prod(shape) for shape in primitive["input_shapes"]) * req["precision"]["input_bitwidth"] // 8
        if (key_path.parent / "input.bin").stat().st_size != input_size:
            raise ValueError("Complete input witness byte count mismatch")
        # Numeric/interface documents are bound by content, not trusted hash labels.
        numeric = _read(_file(key_path.parent, key["numeric_contract"]))
        if canonical_hash(numeric) != req["numeric_contract_hash"]:
            raise ValueError("Numerical contract content mismatch")
        interface = _read(_file(key_path.parent, key["interface_contract"]))
        if req["interface_contract_hash"] is not None:
            if canonical_hash(interface) != req["interface_contract_hash"]:
                raise ValueError("Interface contract content mismatch")
        elif (interface.get("status") != "REVIEWED_BOUNDARY_INTERFACE"
              or interface.get("numeric_contract_hash") != req["numeric_contract_hash"]
              or interface.get("precision") != req["precision"]
              or not all(interface.get(name) for name in ("layout", "framing", "tail_padding"))):
            raise ValueError("Mixed boundary interface is not frozen")
        reference = _file(key_path.parent, dict(path="reference.bin", sha256=key["witness_files"]["reference.bin"]))
        output = _file(self.base, receipt["csim_output"])
        byte_count = math.prod(req["primitive"]["output_shape"]) * req["precision"]["output_bitwidth"] // 8
        if reference.stat().st_size != byte_count or sha(reference) != sha(output):
            raise ValueError("Complete primitive output witness parity/byte count mismatch")
        report = _file(self.base, receipt["csynth_report"])
        metrics = _xml_metrics(report)  # Rechecks actual XML part, tool, target and finite metrics.
        if metrics["cycles"] <= 0 or not 0 < metrics["estimated_clock_ns"] <= PLATFORM["target_clock_ns"]:
            raise ValueError("CSYNTH timing failed")
        csim_log = _file(self.base, receipt["csim_log"])
        hls_log = _file(self.base, receipt["hls_log"])
        if not csim_log.stat().st_size or not hls_log.stat().st_size:
            raise ValueError("Empty tool logs")
        # Bind outputs to the exact executed source key in a result manifest.
        execution = _read(_file(self.base, receipt["execution_record"]))
        expected = dict(task_id=task["task_id"], measurement_key_sha256=sha(key_path),
                        source_hash=key["source_hash"], csynth_report_sha256=sha(report),
                        csim_output_sha256=sha(output), csim_log_sha256=sha(csim_log),
                        hls_log_sha256=sha(hls_log), status="PASS_CSIM_AND_CSYNTH")
        if any(execution.get(name) != value for name, value in expected.items()):
            raise ValueError("Execution/source/report binding mismatch")
        return dict(Latency=metrics["cycles"] * 5 / 1000, LUT=metrics["LUT"], DSP=metrics["DSP"],
                    BRAM=metrics["BRAM18"], source_hash=key["source_hash"], report_sha256=sha(report))

    def query(self, architecture, profile, mapping_policy):
        architecture = validate_architecture(architecture)
        if profile not in PROFILES or mapping_policy not in POLICIES:
            raise ValueError("Unregistered profile or policy")
        identity = (architecture_hash(architecture), profile)
        graph = self.graphs[identity]
        tasks = [self._task(node["primitive_requirement_id"], mapping_policy) for node in graph["nodes"]]
        missing = sorted({task["task_id"] for task in tasks if task["task_id"] not in self.verified})
        reasons = []
        if identity not in self.graph_contracts:
            reasons.append("COMPLETE_INTEGER_GRAPH_CONTRACT_MISSING")
        if any(task["mapping"] is None for task in tasks):
            reasons.append("CONCRETE_MAPPING_NOT_FROZEN")
        if missing:
            reasons.append("PRIMITIVE_SOURCE_VERIFIED_MEASUREMENTS_MISSING")
        result = dict(architecture_id=architecture_id(architecture), architecture_hash=identity[0],
            architecture=deepcopy(architecture), profile=profile, mapping_policy=mapping_policy,
            num_classes=self.num_classes, space_version=SEARCH_SPACE_VERSION,
            topology_sha256=graph["topology_sha256"], status="NEEDS_MEASUREMENT",
            measurement_valid=False, timing_eligible=False, Latency=None, LUT=None, DSP=None, BRAM=None,
            reasons=reasons, missing_task_ids=missing, graph_node_count=len(tasks),
            measured_node_count=sum(task["task_id"] in self.verified for task in tasks),
            evidence_manifest_sha256=self.evidence_sha256, deployment="NOT_CLAIMED",
            latency_scope="HLS_CSYNTH_serial_operator_sum_not_board_e2e",
            resource_scope="peak_kernel_not_integrated_resources_excludes_shell_DDR",
            body8_to_head16_boundary_count=0 if profile == "W8A8" else 4)
        if not reasons:
            values = [self.verified[task["task_id"]] for task in tasks]
            result.update(status="LOOKED_UP", measurement_valid=True, timing_eligible=True,
                          Latency=sum(value["Latency"] for value in values),
                          **{name: max(value[name] for value in values) for name in ("LUT", "DSP", "BRAM")})
        return result


def load_verified_joint_cost_backend(path, *, num_classes, profiles, policies, space_version):
    return VerifiedJointCostBackend(path, num_classes=num_classes, profiles=profiles,
                                    policies=policies, space_version=space_version)


def build_measurement_queue(backend):
    """Deduplicate exact tasks; never launch or turn unknown costs into penalties."""
    tasks = backend._all_tasks()
    for task in tasks.values():
        task.update(status="ACCEPTED" if task["task_id"] in backend.verified else
                    "NEEDS_MAPPING_FREEZE" if task["mapping"] is None else "NEEDS_MEASUREMENT",
                    uses=[], submission_automatic=False)
    rows = []
    for arch in enumerate_architectures():
        for profile in PROFILES:
            graph = backend.graphs[architecture_hash(arch), profile]
            for policy in POLICIES:
                row = backend.query(arch, profile, policy)
                rows.append({key: row[key] for key in ("architecture_id", "profile", "mapping_policy", "status", "reasons")})
                for node in graph["nodes"]:
                    task_id = backend._task(node["primitive_requirement_id"], policy)["task_id"]
                    tasks[task_id]["uses"].append(dict(architecture_id=architecture_id(arch), profile=profile,
                                                       mapping_policy=policy, node_id=node["node_id"]))
    return dict(schema_version=1, manifest_sha256=backend.evidence_sha256,
                triple_count=len(rows), complete_triples=sum(row["status"] == "LOOKED_UP" for row in rows),
                primitive_requirements=len(backend.specs), deduplicated_tasks=len(tasks),
                tasks=list(tasks.values()), triples=rows, hls_launched=False,
                zero_complete_records_does_not_mean_all_triples_measured_failed=True)


def _absolutize(value, base):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return _bind(_file(base, value))
        return {key: _absolutize(item, base) for key, item in value.items()}
    if isinstance(value, list):
        return [_absolutize(item, base) for item in value]
    return value


def import_measurement_results(manifest_path, receipts, output_path):
    """Validate an entire import atomically; original evidence remains untouched.

    receipts is a JSON path containing a list of receipt objects. Re-import of an
    identical task is idempotent; conflicting evidence is rejected, never replaced.
    Mapping plans/graph contracts must already be frozen in the source manifest.
    """
    manifest_path, receipts, output_path = map(Path, (manifest_path, receipts, output_path))
    if output_path.exists():
        raise FileExistsError("Preserve accepted manifest; select a new output path")
    manifest = _absolutize(_read(manifest_path), manifest_path.resolve().parent)
    imported = _absolutize(_read(receipts), receipts.resolve().parent)
    rows = _unique(manifest["measurements"], lambda row: row["task_id"], "measurement")
    for row in imported:
        if row["task_id"] in rows and rows[row["task_id"]] != row:
            raise ValueError("Conflicting result for previously imported task")
        rows[row["task_id"]] = row
    manifest["measurements"] = list(rows.values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".pending")
    if temporary.exists():
        raise FileExistsError("Pending import already exists")
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        backend = load_verified_joint_cost_backend(temporary, num_classes=manifest["num_classes"],
            profiles=PROFILES, policies=POLICIES, space_version=SEARCH_SPACE_VERSION)
        temporary.rename(output_path)
        return dict(status="IMPORTED_SOURCE_VERIFIED", accepted_tasks=len(backend.verified),
                    manifest_path=str(output_path.resolve()), manifest_sha256=sha(output_path), hls_launched=False)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("--manifest", type=Path, required=True)
    queue.add_argument("--output", type=Path, required=True)
    imp = commands.add_parser("import")
    imp.add_argument("--manifest", type=Path, required=True)
    imp.add_argument("--receipts", type=Path, required=True)
    imp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output path")
    if args.command == "import":
        result = import_measurement_results(args.manifest, args.receipts, args.output)
    else:
        result = empty_manifest() if args.command == "init" else build_measurement_queue(
            load_verified_joint_cost_backend(args.manifest, num_classes=2, profiles=PROFILES,
                                             policies=POLICIES, space_version=SEARCH_SPACE_VERSION))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "triple_count", "complete_triples", "deduplicated_tasks")
                      if key in result}))


if __name__ == "__main__":
    main()
