"""Isolated V2 architecture x precision x mapping adapter for the new neck.

The original Pareto and boundary promotion functions execute unchanged.  The
architecture pool, populations and rejected-macro audit are explicitly versioned
changes: 72 unique E0/E1 candidates, 72/54/36 populations, and two macro groups.
This file contains real independent-training/PTQ interfaces, but currently has
no admitted joint hardware backend or frozen common MODNAS decision/GPU budget.
Plan is executable.  Freeze and training fail closed before opening dataset
images until all common-space, source-bound software/hardware gates are met.
No plan, software parity, or callback declaration is a completed search result.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.modnas_elastic.model import (
    SEARCH_SPACE_VERSION, StaticElasticNeckDetector, architecture_hash,
    architecture_id, enumerate_architectures, validate_architecture,
)
from tools.sonar_joint_v2 import policy

VERSION = "sonar_joint_v2_elastic_v1"
CONFIG = ROOT / "configs/search/sonar_joint_v2_elastic_v1.yaml"
DEFAULT_PLAN = ROOT / "results/modnas_vs_ours_20261007/joint_v2_newspace_preflight"
PROFILES, POLICIES = policy.PROFILES, policy.POLICIES
MACROS = ("E0", "E1")


class CampaignBlocked(RuntimeError):
    """A required source-bound execution gate remains unsatisfied."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def architectures():
    pool = {architecture_id(arch): arch for arch in enumerate_architectures()}
    if len(pool) != 72 or sum(not a["pan"] for a in pool.values()) != 8:
        raise RuntimeError("Canonical E0/E1 architecture registry changed")
    return pool


def macro(aid):
    if aid not in architectures():
        raise ValueError("Unknown new-space architecture")
    return "E1" if architectures()[aid]["pan"] else "E0"


def build_training_model(aid, seed=42, num_classes=2):
    """Construct a genuinely independent model; no supernet/checkpoint import."""
    pool = architectures()
    if aid not in pool or seed not in (42, 43, 44):
        raise ValueError("Unknown architecture or unregistered seed")
    return StaticElasticNeckDetector(pool[aid], seed=seed, num_classes=num_classes)


def validate_config(cfg):
    if cfg.get("version") != VERSION or cfg.get("space_version") != SEARCH_SPACE_VERSION:
        raise ValueError("New-space V2 requires its own version and architecture registry")
    if cfg.get("budgets") != [40, 60, 80] or cfg.get("populations") != [72, 54, 36]:
        raise ValueError("Preserve 40/60/80; new-space populations must be 72/54/36")
    if cfg.get("search_seed") not in (42, 43, 44) or cfg.get("confirmation_seeds") != [42, 43, 44]:
        raise ValueError("Independent search seed and internal three-seed confirmation differ")
    if cfg.get("independent_search_seeds_required_for_main_table") != [42, 43, 44]:
        raise ValueError("Main-table three independent searches cannot be replaced by confirmation seeds")
    if cfg.get("quant_profiles") != list(PROFILES) or cfg.get("mapping_policies") != list(POLICIES):
        raise ValueError("All three profiles and all six mapping policies must remain")
    if cfg.get("training") != dict(optimizer="AdamW", lr=.001, weight_decay=.0001,
        scheduler="none", amp=False, workers=0, initialization="independent_from_scratch"):
        raise ValueError("Keep the original V2 independent-training protocol")
    if cfg.get("checkpoint_selection") != "best_valid_mAP50_95_earliest_tie" or cfg.get("shortlist") != 12:
        raise ValueError("V2 checkpoint selection and shortlist changed")
    if cfg.get("calibration") != dict(split="train", images=64, augmentation=False):
        raise ValueError("PTQ requires frozen first-64 train calibration")
    if cfg.get("admission") != dict(absolute_drop=.003, relative_drop=.20, improved_seeds=2):
        raise ValueError("Preserve original V2 paired quantization admission")
    if cfg.get("test_policy") != "sealed" or cfg.get("evaluation_split") != "valid" or cfg.get("deployment_claim") is not False:
        raise ValueError("Validation-only selection; no test/deployment claim")
    dataset = cfg.get("dataset", {})
    if type(dataset.get("num_classes")) is not int or dataset["num_classes"] < 1:
        raise ValueError("Dataset-specific positive class count required")
    baseline = yaml.safe_load(_workspace_file(cfg["baseline"]).read_text(encoding="utf-8"))
    if baseline["dataset"]["num_classes"] != dataset["num_classes"]:
        raise ValueError("Dataset class count differs from its bound training baseline")
    promotion = cfg.get("promotion", {})
    if promotion.get("metric") != "budget_feasible_quantized_mAP50_95_pareto":
        raise ValueError("Original joint Pareto promotion is required")
    for key, maximum in (("fraction", .5), ("accuracy_epsilon", 1), ("resource_fraction", 1)):
        if not policy.finite(promotion.get(key), positive=key == "resource_fraction") or promotion[key] > maximum:
            raise ValueError("Invalid boundary promotion configuration: " + key)
    policy.validate_budget(cfg["hardware_budget"], required=True)
    if cfg["hardware_budget"].get("latency_unit") != "us" or cfg["hardware_budget"].get("bram_unit") != "BRAM18":
        raise ValueError("Preserve operator cost units")
    if cfg.get("rejection_audit") != "one_per_E0_E1_per_stage_to_80_quantized_stop_above_final_top5":
        raise ValueError("The changed E0/E1 macro audit must be explicitly versioned")
    return cfg


def load_config(path=CONFIG):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def triple_requirements(cfg):
    validate_config(cfg)
    rows = []
    for aid, arch in sorted(architectures().items()):
        for profile in cfg["quant_profiles"]:
            for mapping in cfg["mapping_policies"]:
                rows.append(dict(architecture_id=aid, architecture_hash=architecture_hash(arch),
                    architecture=deepcopy(arch), macro=macro(aid), profile=profile,
                    mapping_policy=mapping, num_classes=cfg["dataset"]["num_classes"],
                    space_version=SEARCH_SPACE_VERSION, status="NEEDS_MEASUREMENT",
                    measurement_valid=False, timing_eligible=False,
                    Latency=None, LUT=None, DSP=None, BRAM=None,
                    graph_and_precision_specific_evidence_required=True,
                    full_body_head_and_boundary_cost_required=True,
                    body8_to_head16_boundary_count=0 if profile == "W8A8" else 4,
                    no_precision_or_mapping_alias_allowed=True, deployment="NOT_CLAIMED"))
    if len(rows) != 1296 or len({(r["architecture_id"], r["profile"], r["mapping_policy"]) for r in rows}) != 1296:
        raise RuntimeError("Joint dimension registry is incomplete")
    return rows


def _backend_module(cfg):
    name = cfg["quantization_dispatcher"]
    if name != "tools.modnas_comparison.quantization_mixed":
        raise ValueError("Use the independently versioned new-space PTQ dispatcher")
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


def _workspace_file(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file():
        raise ValueError("Source-bound evidence file missing/outside workspace: " + str(path))
    return target


def _source_proof(path):
    """Validate the supplied artifact and every current executed-source hash."""
    target = _workspace_file(path)
    proof = json.loads(target.read_text(encoding="utf-8"))
    hashes = proof.get("execution_code_sha256", proof.get("source_sha256", {}))
    if not hashes:
        raise ValueError("Backend proof lacks its actually executed source closure")
    for name, expected in hashes.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha(source) != expected:
            raise ValueError("Backend proof source drift: " + name)
    return proof, sha(target)


def _validate_real_software_proof(proof, target, cfg):
    """Verify the real six-result gate, its artifacts and every all-node row.

    No calibration images are opened here. The train-prefix content is rebound
    during freeze and actual preparation; this gate verifies existing evidence.
    """
    if (proof.get("status") != "PASS_REAL_DATASET_A_THREE_PROFILE_SOFTWARE_PTQ_GATE"
            or proof.get("supported_software_profiles") != list(PROFILES)
            or set(proof.get("candidates", {})) != set(MACROS)
            or cfg["dataset"]["id"] != "cylinder2_v6_dataset_a"
            or cfg["dataset"]["num_classes"] != 2):
        raise ValueError("Real Dataset-A E0/E1 three-profile software proof required")
    for key in ("test_image_or_label_files_accessed", "test_evaluated", "accuracy_evaluated",
                "formal_eligible", "HLS_executed", "route_executed", "board_executed"):
        if proof.get(key) is not False:
            raise ValueError("Software proof scope changed: " + key)
    required_sources = {"tools/modnas_comparison/quantization.py",
        "tools/modnas_comparison/quantization_mixed.py",
        "tools/modnas_comparison/real_data_mixed_ptq_gate.py",
        "tools/modnas_elastic/model.py", "tools/modnas_elastic/cost_graph.py",
        "tools/fpga_h2/numeric.py", "tools/fpga_h4/quant_contracts.py"}
    if not required_sources <= set(proof["execution_code_sha256"]):
        raise ValueError("Real PTQ source closure incomplete")
    calibration = deepcopy(proof["calibration"])
    calibration_sha = calibration.pop("calibration_sha256")
    if (digest(calibration) != calibration_sha or calibration["split"] != "train"
            or calibration["number_of_calibration_images"] != 64
            or calibration["input_chw"] != [1, 640, 640]
            or len(calibration["entries"]) != 64
            or any(calibration.get(k) is not False for k in
                   ("augmentation", "shuffle", "valid_used_for_selection", "test_accessed", "test_evaluated"))):
        raise ValueError("Real software gate calibration binding changed")
    names = [r["image"] for r in calibration["entries"]]
    if names != sorted(set(names)) or any(not n.replace("\\", "/").startswith("train/images/") for n in names):
        raise ValueError("Software gate calibration is not sorted first64 train")
    source_directory = (ROOT / proof["source_directory"]).resolve()
    if not source_directory.is_relative_to(ROOT):
        raise ValueError("Software checkpoint evidence outside workspace")
    for name, expected in proof["source_artifact_sha256"].items():
        if sha(_workspace_file(source_directory / name)) != expected:
            raise ValueError("Software checkpoint artifact drift: " + name)
    if proof["source_checkpoint_execution_manifest_sha256"] != proof["source_artifact_sha256"].get("execution_manifest.json"):
        raise ValueError("Software checkpoint execution manifest mismatch")
    artifact_count, total_nodes, mixed_boundaries = 0, 0, []
    for group in MACROS:
        candidate = proof["candidates"][group]
        arch = validate_architecture(candidate["architecture"])
        aid = architecture_id(arch)
        if (candidate["architecture_id"] != aid or macro(aid) != group
                or candidate["architecture_sha256"] != architecture_hash(arch)
                or set(candidate["profiles"]) != set(PROFILES)
                or candidate["source_checkpoint_sha256"] != proof["source_artifact_sha256"].get(candidate["source_checkpoint"])):
            raise ValueError("Software candidate architecture/checkpoint binding mismatch")
        for profile in PROFILES:
            summary = candidate["profiles"][profile]
            leaf = target.parent / group / profile
            required_files = {"artifact_manifest.json", "extrema.json", "integer_parameters.pt",
                              "numeric_graph.json", "profile.json", "software_parity_witness.json"}
            hashes = summary["artifact_sha256"]
            if set(hashes) != required_files:
                raise ValueError("PTQ evidence leaf manifest incomplete")
            for name, expected in hashes.items():
                if sha(_workspace_file(leaf / name)) != expected:
                    raise ValueError("PTQ artifact drift: " + group + "/" + profile + "/" + name)
                artifact_count += 1
            manifest = json.loads((leaf / "artifact_manifest.json").read_text(encoding="utf-8"))
            if manifest["files"] != {k: v for k, v in hashes.items() if k != "artifact_manifest.json"}:
                raise ValueError("PTQ nested artifact manifest disagrees with real gate")
            record = json.loads((leaf / "profile.json").read_text(encoding="utf-8"))
            graph = json.loads((leaf / "numeric_graph.json").read_text(encoding="utf-8"))
            witness = json.loads((leaf / "software_parity_witness.json").read_text(encoding="utf-8"))
            graph_body = deepcopy(graph)
            graph_sha = graph_body.pop("numeric_graph_sha256")
            if digest(graph_body) != graph_sha:
                raise ValueError("PTQ numeric graph semantic digest changed")
            if (record["precision_profile"] != profile or record["num_classes"] != 2
                    or record["architecture"] != arch or graph["architecture"] != arch
                    or graph["num_classes"] != 2 or graph["architecture_sha256"] != architecture_hash(arch)
                    or record["calibration"] != proof["calibration"]
                    or summary["actual_train_calibration_images"] != 64
                    or summary["integer_witness_train_images"] != 1 or witness["input_images"] != 1):
                raise ValueError("PTQ profile architecture/class/calibration identity changed")
            for value in (record["original_model_state_sha256"], witness["original_model_state_sha256"], summary["source_state_sha256"]):
                if value != candidate["source_state_sha256"]:
                    raise ValueError("PTQ original source-state hash changed")
            for key in ("numeric_graph_sha256", "integer_parameters_sha256"):
                if not (record[key] == summary[key] == witness[key]):
                    raise ValueError("PTQ witness numeric binding mismatch: " + key)
            if (record["numeric_contract_hash"] != summary["numeric_contract_hash"]
                    or witness["calibration_sha256"] != calibration_sha
                    or witness["status"] != "PASS_SOFTWARE_INTEGER_FAKE_PARITY"
                    or summary["software_parity_status"] != witness["status"]
                    or witness["mismatches"] != 0 or summary["actual_mismatch_count"] != 0
                    or witness.get("test_evaluated") is not False):
                raise ValueError("PTQ all-node zero-mismatch witness incomplete")
            rows = witness["rows"]
            expected_nodes = {n["id"] for n in graph["nodes"]} | {"image"}
            if (len(rows) != len(expected_nodes) or {r["node"] for r in rows} != expected_nodes
                    or summary["compared_nodes"] != len(rows)
                    or summary["compared_elements"] != sum(r["element_count"] for r in rows)
                    or any(r["mismatch_count"] != 0 or r["integer_sha256"] != r["fake_quantized_sha256"]
                           or not policy.finite(r["element_count"], positive=True) for r in rows)):
                raise ValueError("Real all-node witness coverage or per-node hash parity failed")
            if profile != "W8A8":
                boundaries = record["head_input_conversions"]
                if (len(boundaries) != 4 or {n["id"] for n in boundaries} !=
                        {f"head.towers.{i}.input_requant" for i in range(4)}
                        or not {n["id"] for n in boundaries} <= expected_nodes):
                    raise ValueError("Mixed-head four requantization boundaries missing")
                mixed_boundaries.append(dict(macro=group, profile=profile, count=4,
                    hardware_status="UNVERIFIED_REQUIRES_PRECISION_SPECIFIC_MEASUREMENT"))
            evidence = proof["software_profile_evidence"][profile]
            if evidence["real_DatasetA_E0_E1_all_node_parity"] != "PASS" or evidence["total_mismatches"] != 0:
                raise ValueError("Real software aggregate differs from six witnesses")
            total_nodes += len(rows)
    return dict(verified_artifacts=artifact_count, verified_execution_sources=len(proof["execution_code_sha256"]),
                verified_all_node_rows=total_nodes, macro_profile_groups=6,
                calibration_images=64, integer_witness_images_per_group=1,
                mixed_head_boundary_requirements=mixed_boundaries,
                accuracy_retention="NOT_EVALUATED", hardware_feasibility="UNKNOWN")


def software_readiness(cfg):
    evidence = cfg.get("software_backend_evidence")
    proof, proof_sha = _source_proof(evidence) if evidence else (None, None)
    if evidence and proof_sha != cfg.get("software_backend_evidence_sha256"):
        raise ValueError("Software gate file differs from preregistered proof SHA256")
    # Verify executed code before importing its preparation runtime.
    module = _backend_module(cfg)
    supported = tuple(getattr(module, "SUPPORTED_PROFILES", ())) if module else ()
    missing = [profile for profile in PROFILES if profile not in supported]
    result = dict(module=cfg["quantization_dispatcher"], declared_supported_profiles=list(supported),
                  missing_profiles=missing, status="BLOCKED", proof_sha256=None)
    if missing:
        result["reason"] = "UNSUPPORTED_NEWSPACE_PRECISION_PROFILES"
        return result
    if not evidence:
        result["reason"] = "SOURCE_BOUND_ALL_PROFILE_SOFTWARE_PARITY_EVIDENCE_MISSING"
        return result
    details = _validate_real_software_proof(proof, _workspace_file(evidence), cfg)
    result.update(status="READY", scope="SOURCE_BOUND_DATASET_A_E0_E1_SOFTWARE_BACKEND_PARITY_ONLY",
                  proof_sha256=proof_sha, evidence=evidence, **details)
    return result


def _cost_backend(cfg):
    """A real source-verifying joint cost backend is not present in this adapter.

    A future reviewed backend must enumerate precision- and mapping-specific
    source-validated measurements; a JSON flag or copied W8A8 row is insufficient.
    It is deliberately not replaced by FLOPs, Params, aliases or fake mapping.
    """
    name = cfg.get("hardware_cost_backend")
    if name is None:
        return None
    if name != "tools.modnas_comparison.joint_cost_backend":
        raise ValueError("Unregistered new-space joint hardware cost backend")
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            raise CampaignBlocked("NEWSPACE_JOINT_HARDWARE_BACKEND_NOT_IMPLEMENTED") from error
        raise
    backend = module.load_verified_joint_cost_backend(ROOT / cfg["hardware_cost_evidence"],
        num_classes=cfg["dataset"]["num_classes"], profiles=PROFILES, policies=POLICIES,
        space_version=SEARCH_SPACE_VERSION)
    # This API is a future dependency contract, not a claim that the backend exists.
    return backend


def preflight(cfg):
    """Enumerate demand and inspect gates without decoding any train/valid/test image."""
    requirements = triple_requirements(cfg)
    software = software_readiness(cfg)
    blockers = []
    if software["status"] != "READY":
        blockers.append(software["reason"])
    backend = _cost_backend(cfg)
    costs = []
    for requirement in requirements:
        if backend is None:
            costs.append(requirement)
            continue
        measured = backend.query(requirement["architecture"], requirement["profile"], requirement["mapping_policy"])
        if any(measured.get(key) != requirement[key] for key in
               ("architecture_id", "architecture_hash", "profile", "mapping_policy", "num_classes", "space_version")):
            raise ValueError("Joint hardware response architecture/profile/mapping/class binding mismatch")
        costs.append(measured)
    screened = policy.filter_costs(costs, cfg["hardware_budget"])
    if backend is None:
        blockers.append("NEWSPACE_JOINT_HARDWARE_BACKEND_NOT_IMPLEMENTED")
    qualified = sum(row.get("status") == "LOOKED_UP" and row.get("measurement_valid") is True
                    and row.get("timing_eligible") is True
                    and all(policy.finite(row.get(metric)) for metric in policy.METRICS) for row in costs)
    if qualified != len(requirements):
        blockers.append("INCOMPLETE_SOURCE_VERIFIED_1296_HARDWARE_TRIPLES")
    if not screened["rows"]:
        blockers.append("NO_HARDWARE_BUDGET_FEASIBLE_TRIPLES")
    comparison = cfg["comparison"]
    if (not isinstance(comparison.get("gpu_model"), str) or not comparison["gpu_model"]
            or not policy.finite(comparison.get("gpu_h_cap_per_method_per_independent_search"), positive=True)):
        blockers.append("COMMON_GPU_MODEL_AND_TOTAL_BUDGET_NOT_FROZEN")
    if (comparison.get("common_decision_space_status") != "FROZEN"
            or comparison.get("modnas_decision_interface_status") not in
            ("VERIFIED_SHARED_1296_DECISIONS", "VERIFIED_SHARED_POST_EVALUATION_PROTOCOL")):
        blockers.append("MODNAS_COMMON_PROFILE_MAPPING_DECISION_PROTOCOL_UNIMPLEMENTED")
    if cfg["dataset"].get("split_status") != "frozen_train_valid":
        blockers.append("DATASET_SPLIT_NOT_FROZEN")
    blockers.append("COMMON_GPU_ACCOUNTING_LAUNCH_DRIVER_NOT_IMPLEMENTED")
    return dict(version=VERSION, status="BLOCKED_NOT_LAUNCHABLE" if blockers else "READY_FOR_FREEZE",
        blockers=blockers, architecture_count=72, macros={"E0": 8, "E1": 64}, triple_count=1296,
        cost_rows=costs, qualified_hardware_triples=qualified,
        eligible_triples=len(screened["rows"]), eligible_architectures=sorted({r["architecture_id"] for r in screened["rows"]}),
        software_backend=software, hardware_cost_backend=cfg.get("hardware_cost_backend"),
        hardware_budget=cfg["hardware_budget"], common_decision_space="UNFROZEN_REQUIREMENT_JOINT_ENCODING_ONLY",
        dataset_images_opened=False, training_started=False, test_accessed=False, test_evaluated=False,
        deployment="NOT_CLAIMED")


def _maximum_work(cfg):
    # Progressive training reuses its own earlier-budget optimizer/RNG, not an
    # unrelated old architecture checkpoint. Two E0/E1 audits at each rejection
    # stage give at most four audit candidates, each eventually trained to 80.
    training = 72 * 40 + 54 * 20 + 36 * 20 + 4 * 80 + 12 * 2 * 80
    return dict(max_training_epoch_equivalents_per_independent_search=training,
        max_stage_ptq_profile_evaluations=(72 + 54 + 36 + 4 + 12 * 2) * 3,
        numerical_gpu_budget=None, epoch_equivalents_are_not_GPU_hours=True)


def plan(config_path=CONFIG, output=DEFAULT_PLAN):
    cfg = load_config(config_path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve earlier plans; select a new empty output directory")
    report = preflight(cfg)
    report.update(created_utc=datetime.now(timezone.utc).isoformat(), protocol=cfg,
        configuration_sha256=sha(config_path), adapter_sha256=sha(Path(__file__)),
        reused_original_policy_sha256=sha(ROOT / "tools/sonar_joint_v2/policy.py"),
        changed_contract=["old144 backbone pool -> canonical72 neck pool",
            "populations144/108/72 -> 72/54/36", "three old macros -> E0/E1 rejection audit"],
        preserved_mechanisms=["40/60/80 progressive independent training", "three PTQ profiles",
            "six hardware mapping policies", "unchanged Pareto/boundary promotion",
            "three-seed paired confirmation distinct from independent search repetitions"],
        implemented_interfaces=["complete1296 requirement enumeration", "source/gate preflight",
            "new independent training model", "unchanged policy promotion", "E0/E1 rejection selection",
            "unchanged original paired confirmation/admission", "guarded real training/resume",
            "guarded real calibration/full-valid PTQ interface"],
        missing_execution_dependencies=report["blockers"],
        implemented_but_unexecuted=["independent training/resume", "checkpoint-bound full-valid PTQ", "three-seed confirmation interface"],
        full_comparison_adapter_complete=False, **_maximum_work(cfg))
    save(output / "plan.json", report)
    save(output / "requirements_1296.json", triple_requirements(cfg))
    return report


def _source_inputs(config_path, cfg):
    from tools.modnas_elastic.protocol import source_binding
    sources = source_binding()
    paths = [Path(__file__), Path(config_path), ROOT / "tools/sonar_joint_v2/policy.py"]
    paths.extend((ROOT / "tools/sonar_spec").glob("*.py"))
    paths.extend((ROOT / "tools/modnas_comparison").glob("*.py"))
    paths.extend([ROOT / "tools/sonar_joint_v2/run.py", ROOT / "tools/sonar_search/run.py"])
    paths.extend((ROOT / "tools/fpga_h4").glob("*.py"))
    paths.extend((ROOT / "tools/sonar_search_finalize").glob("*.py"))
    for field in ("software_backend_evidence", "hardware_cost_evidence"):
        if cfg.get(field):
            paths.append(ROOT / cfg[field])
    for path in paths:
        relative = path.resolve().relative_to(ROOT).as_posix()
        sources[relative] = sha(path)
    return dict(sorted(sources.items()))


def _verify_freeze_sources(binding):
    """Require the complete current dependency set, not a deletable subset."""
    cfg = validate_config(binding["protocol"])
    config_path = _workspace_file(binding["config_path"])
    if (binding["config_sha256"] != sha(config_path)
            or binding["protocol_sha256"] != digest(cfg)
            or load_config(config_path) != cfg
            or binding["source_closure_sha256"] != digest(binding["source_sha256"])):
        raise ValueError("Frozen configuration/source closure digest mismatch")
    if _source_inputs(config_path, cfg) != binding["source_sha256"]:
        raise ValueError("Frozen complete execution-source closure changed or lost required keys")
    return cfg


def freeze(config_path, output):
    """Reject blocked hardware/software/comparison gates before reading data."""
    cfg = load_config(config_path)
    report = preflight(cfg)
    if report["status"] != "READY_FOR_FREEZE":
        raise CampaignBlocked("; ".join(report["blockers"]))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Freeze requires a new empty campaign directory")
    from tools.modnas_elastic.protocol import dataset_binding
    baseline = yaml.safe_load((ROOT / cfg["baseline"]).read_text(encoding="utf-8"))
    data = dataset_binding(baseline)
    sources = _source_inputs(config_path, cfg)
    binding = dict(version=VERSION, protocol=cfg, protocol_sha256=digest(cfg),
        config_path=Path(config_path).resolve().relative_to(ROOT).as_posix(), config_sha256=sha(config_path),
        source_sha256=sources, source_closure_sha256=digest(sources),
        dataset={split: item["entries"] for split, item in data.items()},
        costs=report["cost_rows"], costs_sha256=digest(report["cost_rows"]),
        search_seed=cfg["search_seed"], independent_search_run_id=f"{VERSION}_searchseed{cfg['search_seed']}",
        initialization="independent_newspace_from_scratch", test_accessed=False,
        test_evaluated=False, deployment="NOT_CLAIMED", training_started=False)
    save(output / "freeze.json", binding)
    return binding


def verify_freeze(output):
    """Current readiness/source checks precede dataset-content verification."""
    output = Path(output)
    binding = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    cfg = _verify_freeze_sources(binding)
    report = preflight(cfg)
    if report["status"] != "READY_FOR_FREEZE":
        raise CampaignBlocked("; ".join(report["blockers"]))
    if binding.get("version") != VERSION or set(binding.get("dataset", {})) != {"train", "valid"}:
        raise ValueError("Newspace freeze identity or split violation")
    if (binding.get("test_evaluated") is not False or binding.get("test_accessed") is not False
            or binding.get("costs_sha256") != digest(binding["costs"])):
        raise ValueError("Frozen test/cost binding mismatch")
    if report["cost_rows"] != binding["costs"]:
        raise ValueError("Frozen source-verified hardware costs changed")
    from tools.modnas_elastic.protocol import dataset_binding
    baseline = yaml.safe_load((ROOT / cfg["baseline"]).read_text(encoding="utf-8"))
    actual = dataset_binding(baseline)
    if {split: item["entries"] for split, item in actual.items()} != binding["dataset"]:
        raise ValueError("Frozen train/valid contents changed")
    return binding, sha(output / "freeze.json"), baseline


def select_rejection_audits(previous, selected, *, stage, search_seed):
    if stage not in (0, 1) or search_seed not in (42, 43, 44):
        raise ValueError("Only two rejected stages and registered search seeds")
    if len(set(previous)) != len(previous) or not set(selected).issubset(previous):
        raise ValueError("Invalid stage populations")
    removed = set(previous) - set(selected)
    result = []
    for group in MACROS:
        pool = [aid for aid in removed if macro(aid) == group]
        if pool:
            result.append(min(pool, key=lambda aid: digest([search_seed, stage, aid])))
    return result


def promote_stage(evaluations, costs, cfg, *, stage, previous):
    """Execute the original V2 joint_points/promote on this adapter's identities."""
    validate_config(cfg)
    if stage not in (0, 1, 2) or set(previous) - set(architectures()):
        raise ValueError("Invalid newspace promotion stage")
    if len(previous) != len(set(previous)) or {r["architecture_id"] for r in evaluations} != set(previous):
        raise ValueError("Each staged architecture needs exactly one actual evaluation")
    for result in evaluations:
        if (result["seed"] != cfg["search_seed"] or result["budget"] != cfg["budgets"][stage]
                or set(result["profiles"]) != set(PROFILES)):
            raise ValueError("Stage evaluation seed/budget/precision mismatch")
    points = policy.joint_points(evaluations, costs, cfg["hardware_budget"])
    count = cfg["populations"][stage + 1] if stage < 2 else cfg["shortlist"]
    promoted = policy.promote(points, count, cfg["promotion"])
    if not promoted["selected"]:
        raise CampaignBlocked("No actual quantized budget-feasible promotion points")
    promoted["rejection_audits"] = select_rejection_audits(previous, promoted["selected"],
        stage=stage, search_seed=cfg["search_seed"]) if stage < 2 else []
    promoted["macro_contract"] = "new_E0_E1_not_old_three_backbone_macros"
    return promoted


def check_rejection_audit(final_points, audit_points):
    best = sorted((max(p["Accuracy"] for p in final_points if p["architecture_id"] == aid)
                   for aid in {p["architecture_id"] for p in final_points}), reverse=True)
    if not best:
        raise ValueError("Actual final promoted points required before rejection audit")
    threshold = best[min(4, len(best) - 1)]
    failures = sorted({p["architecture_id"] for p in audit_points if p["Accuracy"] > threshold})
    result = dict(quantized_top5_threshold=threshold, violations=failures,
                  test_evaluated=False, scope="quantized_valid_rejection_audit")
    if failures:
        raise CampaignBlocked("Rejected E0/E1 audit exceeded promoted top5: " + ",".join(failures))
    return result


def confirmed_points(evaluations, costs, cfg):
    """Run unchanged V2 paired admission/confirmation on new architecture IDs.

    This pure policy interface is unit-tested with explicit synthetic fixtures.
    It is not evidence that any three actual searches or confirmations ran.
    """
    validate_config(cfg)
    if not evaluations or set(evaluations) - set(architectures()):
        raise ValueError("Actual newspace confirmation candidates required")
    from tools.sonar_joint_v2.run import confirmed_points as original_confirmed_points
    result = original_confirmed_points(evaluations, costs, cfg)
    result.update(independent_search_seed=cfg["search_seed"],
                  n_independent_searches_represented=1,
                  confirmation_seeds_are_not_independent_search_repetitions=True)
    return result


def task_requests(cfg, ids, budget, seed):
    validate_config(cfg)
    if budget not in cfg["budgets"] or seed not in cfg["confirmation_seeds"]:
        raise ValueError("Unregistered task budget/seed")
    if seed != cfg["search_seed"] and budget != 80:
        raise ValueError("Internal confirmation seeds are full-budget only")
    if len(ids) != len(set(ids)) or set(ids) - set(architectures()):
        raise ValueError("Newspace unique candidate IDs required")
    return [dict(architecture_id=aid, architecture=deepcopy(architectures()[aid]),
        num_classes=cfg["dataset"]["num_classes"], seed=seed, budget=budget,
        profiles=list(PROFILES), mapping_policies=list(POLICIES),
        train_split="train", architecture_selection_split="valid", test_evaluated=False)
        for aid in ids]


def _runtime_helpers():
    # The retained train_epoch/optimizer-RNG helper is used with the new model.
    sys.path[:0] = [str(ROOT / "tools/sonar_spec"), str(ROOT / "tools/sonar_analysis"), str(ROOT / "scripts")]
    import baseline as frozen
    from formal_runner import train_epoch, atomic_save, rng_state, restore_rng
    from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
    return frozen, train_epoch, atomic_save, rng_state, restore_rng, DenseAnchorDetectionLoss


def _validate_training_state(state, aid, seed, budget, freeze_sha, num_classes):
    expected = dict(freeze_sha256=freeze_sha, architecture_id=aid, seed=seed,
                    architecture=architectures()[aid], num_classes=num_classes, test_evaluated=False)
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("Newspace training resume identity mismatch")
    history = state["history"]
    if (not history or type(state["epoch"]) is not int or state["epoch"] != len(history)
            or len(history) > budget or [r["epoch"] for r in history] != list(range(1, len(history)+1))):
        raise ValueError("Training resume epoch/history budget mismatch")
    scores = [policy.score(r["valid"]["mAP50_95"]) for r in history]
    best_score = max(scores)
    if state["best_epoch"] != scores.index(best_score)+1 or state["best_score"] != best_score:
        raise ValueError("Training resume best state does not match earliest historical valid maximum")
    if not state["best_state"] or set(state["best_state"]) != set(state["model_state_dict"]):
        raise ValueError("Training resume best tensor set incomplete")
    for name, tensor in state["best_state"].items():
        current = state["model_state_dict"][name]
        if not torch.is_tensor(tensor) or tensor.shape != current.shape or tensor.dtype != current.dtype or not torch.isfinite(tensor).all():
            raise ValueError("Training resume best tensor shape/dtype/finite mismatch")
    if not state.get("optimizer_state_dict") or not state.get("rng"):
        raise ValueError("Training resume optimizer/RNG state missing")


def _validate_budget_checkpoint(state, receipt, aid, seed, budget, freeze_sha, num_classes):
    expected = dict(freeze_sha256=freeze_sha, architecture_id=aid, seed=seed, budget=budget,
                    architecture=architectures()[aid], num_classes=num_classes, test_evaluated=False)
    if any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("PTQ newspace checkpoint identity mismatch")
    if any(receipt.get(k) != v for k, v in expected.items() if k != "architecture"):
        raise ValueError("Training task receipt identity mismatch")
    if (type(state["epoch"]) is not int or not 1 <= state["epoch"] <= budget
            or receipt["best_epoch"] != state["epoch"]
            or receipt["mAP50_95"] != policy.score(state["valid"]["mAP50_95"])):
        raise ValueError("Training task checkpoint epoch/validation receipt mismatch")


def _training_receipt(output, aid, seed, budget, freeze_sha, num_classes):
    folder = Path(output)
    receipt_path = folder / "tasks" / f"train_{aid}_{seed}_{budget}.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = folder / "training" / aid / f"seed_{seed}" / f"best_{budget}.pt"
    if (receipt["checkpoint"] != checkpoint.relative_to(folder).as_posix()
            or receipt["checkpoint_sha256"] != sha(checkpoint)):
        raise ValueError("Training receipt checkpoint path or SHA256 drift")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _validate_budget_checkpoint(state, receipt, aid, seed, budget, freeze_sha, num_classes)
    return receipt, state, checkpoint


def _validate_ptq_witness(prepared, witness, *, profile, architecture, num_classes,
                          source_state_sha, calibration_sha):
    record = prepared.profile
    graph = prepared.integer_reference.graph
    if (witness.get("status") != "PASS_SOFTWARE_INTEGER_FAKE_PARITY"
            or witness.get("mismatches") != 0 or witness.get("test_evaluated") is not False
            or witness.get("input_images") != 1):
        raise ValueError("Actual integer parity failed before PTQ save or accuracy evaluation")
    if (record["precision_profile"] != profile or record["architecture"] != architecture
            or record["num_classes"] != num_classes or graph["architecture"] != architecture
            or graph["num_classes"] != num_classes
            or record["original_model_state_sha256"] != source_state_sha
            or witness["original_model_state_sha256"] != source_state_sha
            or record["calibration"]["calibration_sha256"] != calibration_sha
            or witness["calibration_sha256"] != calibration_sha):
        raise ValueError("PTQ witness architecture/class/source/calibration binding mismatch")
    graph_body = deepcopy(graph)
    graph_sha = graph_body.pop("numeric_graph_sha256")
    if (digest(graph_body) != graph_sha or not
            graph_sha == record["numeric_graph_sha256"] == witness["numeric_graph_sha256"]
            or record["integer_parameters_sha256"] != witness["integer_parameters_sha256"]
            or witness.get("precision_profile", profile) != profile):
        raise ValueError("PTQ witness graph/parameters/profile binding mismatch")
    expected_nodes = {n["id"] for n in graph["nodes"]} | {"image"}
    rows = witness["rows"]
    if (len(rows) != len(expected_nodes) or {r["node"] for r in rows} != expected_nodes
            or any(r["mismatch_count"] != 0 or r["integer_sha256"] != r["fake_quantized_sha256"] for r in rows)):
        raise ValueError("PTQ witness must cover every real node with zero hash mismatches")


def train_candidate(output, aid, seed, budget, *, device="cuda"):
    """Real epoch-boundary resumable independent training; never called by plan."""
    binding, freeze_sha, baseline = verify_freeze(output)
    cfg = binding["protocol"]
    task_requests(cfg, [aid], budget, seed)
    completed = _training_receipt(output, aid, seed, budget, freeze_sha, cfg["dataset"]["num_classes"])
    if completed is not None:
        return completed[0]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU fallback")
    frozen, train_epoch, atomic_save, rng_state, restore_rng, criterion_type = _runtime_helpers()
    frozen.seed_all(seed)
    model = build_training_model(aid, seed, cfg["dataset"]["num_classes"]).to(device)
    # Reset after model construction so candidate shapes do not shift augmentation RNG.
    frozen.seed_all(seed)
    data = ROOT / baseline["dataset"]["data_dir"]
    train = frozen.make_loader(baseline, data, "train", seed, 0, train=True)
    valid = frozen.make_loader(baseline, data, "valid", seed, 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    criterion = criterion_type(model.loss_config(baseline)).to(device)
    folder = Path(output) / "training" / aid / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    latest = folder / "latest.pt"
    history, best, best_epoch, best_score = [], None, None, -1.0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        _validate_training_state(state, aid, seed, budget, freeze_sha, cfg["dataset"]["num_classes"])
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        restore_rng(state["rng"], train)
        history, best, best_epoch, best_score = state["history"], state["best_state"], state["best_epoch"], state["best_score"]
    for epoch in range(len(history) + 1, budget + 1):
        start = time.monotonic()
        row = train_epoch(model, optimizer, train, baseline, device, None)
        metrics, _ = frozen.evaluate(model, valid, criterion, baseline, device)
        policy.score(metrics["mAP50_95"])
        row.update(epoch=epoch, valid=metrics, wall_seconds=time.monotonic() - start)
        history.append(row)
        if metrics["mAP50_95"] > best_score:
            best_score, best_epoch = metrics["mAP50_95"], epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        atomic_save(latest, dict(freeze_sha256=freeze_sha, architecture_id=aid,
            architecture=architectures()[aid], seed=seed, epoch=epoch, history=history,
            num_classes=cfg["dataset"]["num_classes"],
            model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            rng=rng_state(train), best_state=best, best_score=best_score, best_epoch=best_epoch,
            test_evaluated=False))
    if best is None:
        raise ValueError("No completed epoch")
    checkpoint = folder / f"best_{budget}.pt"
    atomic_save(checkpoint, dict(freeze_sha256=freeze_sha, architecture_id=aid,
        architecture=architectures()[aid], seed=seed, epoch=best_epoch, budget=budget,
        num_classes=cfg["dataset"]["num_classes"], model_state_dict=best,
        valid=history[best_epoch - 1]["valid"], test_evaluated=False))
    result = dict(architecture_id=aid, seed=seed, budget=budget, best_epoch=best_epoch,
        num_classes=cfg["dataset"]["num_classes"],
        mAP50_95=best_score, checkpoint=checkpoint.relative_to(output).as_posix(),
        checkpoint_sha256=sha(checkpoint), freeze_sha256=freeze_sha, test_evaluated=False)
    save(Path(output) / "tasks" / f"train_{aid}_{seed}_{budget}.json", result)
    return result


def quant_candidate(output, aid, seed, budget, *, device="cuda"):
    """Real train64 preparation and full-valid PTQ using the new dispatcher.

    This interface cannot execute while backend/common hardware/budget gates are
    missing.  The verification barrier occurs before calibration decodes data.
    """
    binding, freeze_sha, baseline = verify_freeze(output)
    cfg = binding["protocol"]
    task_requests(cfg, [aid], budget, seed)
    module = _backend_module(cfg)
    completed = _training_receipt(output, aid, seed, budget, freeze_sha, cfg["dataset"]["num_classes"])
    if completed is None:
        raise ValueError("PTQ requires completed hash-bound training task receipt")
    receipt, state, checkpoint = completed
    checkpoint_sha = receipt["checkpoint_sha256"]
    model = build_training_model(aid, seed, cfg["dataset"]["num_classes"]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    source_state_sha = module.model_state_hash(model)
    frozen, _, _, _, _, criterion_type = _runtime_helpers()
    data = ROOT / baseline["dataset"]["data_dir"]
    calibration = module.TrainCalibration(data, binding["dataset"]["train"], num_images=64, formal=True)
    valid = frozen.make_loader(baseline, data, "valid", seed, 0)
    criterion = criterion_type(model.loss_config(baseline)).to(device)
    evaluate = lambda candidate: frozen.evaluate(candidate, valid, criterion, baseline, device)[0]
    result = dict(architecture_id=aid, seed=seed, budget=budget, split="valid",
        checkpoint_sha256=checkpoint_sha, freeze_sha256=freeze_sha,
        calibration_sha256=calibration.manifest["calibration_sha256"], test_evaluated=False,
        fp32=evaluate(model), profiles={})
    folder = Path(output) / "quantization" / aid / f"seed_{seed}" / f"budget_{budget}"
    prepared_profiles = module.prepare_elastic_profiles(model, calibration, device=device)
    for profile in PROFILES:
        prepared = prepared_profiles[profile]
        prepared.profile["budget_checkpoint_binding"] = dict(architecture_id=aid,
            architecture_sha256=architecture_hash(architectures()[aid]), num_classes=cfg["dataset"]["num_classes"],
            seed=seed, budget=budget, freeze_sha256=freeze_sha, checkpoint_sha256=checkpoint_sha,
            calibration_sha256=calibration.manifest["calibration_sha256"], split="valid", test_evaluated=False)
        witness = module.verify_fake_integer_parity(prepared, next(calibration.batches(device)))
        _validate_ptq_witness(prepared, witness, profile=profile, architecture=architectures()[aid],
            num_classes=cfg["dataset"]["num_classes"], source_state_sha=source_state_sha,
            calibration_sha=calibration.manifest["calibration_sha256"])
        artifact_folder = module.save_ptq_artifacts(folder / profile, prepared, witness=witness)
        metrics = evaluate(prepared.fake_model)
        policy.score(metrics["mAP50_95"])
        profile_path = folder / profile / "profile.json"
        result["profiles"][profile] = dict(metrics, profile_sha256=sha(profile_path),
            integer_parity_witness=witness,
            artifact_manifest=(Path(artifact_folder) / "artifact_manifest.json").relative_to(output).as_posix(),
            artifact_manifest_sha256=sha(Path(artifact_folder) / "artifact_manifest.json"))
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint changed during PTQ")
    save(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "freeze"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    result = plan(args.config, args.output) if args.mode == "plan" else freeze(args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "architecture_count", "triple_count", "blockers") if key in result}, indent=2))


if __name__ == "__main__":
    main()
