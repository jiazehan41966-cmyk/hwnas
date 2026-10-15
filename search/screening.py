"""Validated screening adapter over the retained multi-fidelity Pareto policy."""
from pathlib import Path
import math
import json

from experiment import validate_record, sha_file
from space import Genotype, canonical_hash
from quant import QuantizationPlan
from . import policy

METRICS = {"Latency":"latency_us", "LUT":"lut", "DSP":"dsp", "BRAM":"bram18"}


def _verify_artifacts(record, *, require_hardware=True):
    required = {"checkpoint", "quant_artifact", "calibration", "dataset", "quantization_result"}
    if require_hardware:
        required.add("hardware_evidence")
    artifacts = record.get("artifacts", {})
    if not required <= set(artifacts):
        raise ValueError("Scored candidates require actual retained artifacts")
    for name, item in artifacts.items():
        if sha_file(item["path"]) != item["sha256"]:
            raise ValueError("Candidate artifact drift: " + name)
    for name, field in (("checkpoint","checkpoint_sha256"),("quant_artifact","quant_artifact_sha256"),
                        ("calibration","calibration_sha256"),("dataset","dataset_sha256")):
        if artifacts[name]["sha256"] != record["accuracy"][field]:
            raise ValueError("Accuracy artifact identity mismatch: " + name)
    from quant.executor import verify_saved_artifacts
    verify_saved_artifacts(Path(artifacts["quant_artifact"]["path"]).parent,
                           expected_arch_hash=record["arch_hash"],expected_q_hash=record["q_hash"])
    checkpoint = Path(artifacts["checkpoint"]["path"])
    sidecar = json.loads(checkpoint.with_suffix(".binding.json").read_text(encoding="utf-8"))
    binding = sidecar["binding"]
    if (sidecar["checkpoint_sha256"] != record["accuracy"]["checkpoint_sha256"]
            or binding["arch_hash"] != record["arch_hash"] or binding["genotype"] != record["genotype"]
            or binding["seed"] != record["seed"] or binding["budget"] != record["budget"]):
        raise ValueError("Checkpoint belongs to a different genotype/seed/budget")
    folder = Path(artifacts["quant_artifact"]["path"]).parent
    profile = json.loads((folder/"profile.json").read_text(encoding="utf-8"))
    calibration = json.loads(Path(artifacts["calibration"]["path"]).read_text(encoding="utf-8"))
    datasets = json.loads(Path(artifacts["dataset"]["path"]).read_text(encoding="utf-8"))
    if profile["calibration"] != calibration or canonical_hash(datasets) != binding["dataset_sha256"]:
        raise ValueError("PTQ calibration/dataset differs from the bound training inputs")
    from train.training import load_bound_checkpoint
    from quant.executor import model_state_hash
    from types import SimpleNamespace
    state = load_bound_checkpoint(checkpoint,binding)
    if model_state_hash(SimpleNamespace(state_dict=lambda:state["model_state_dict"])) != profile["original_model_state_sha256"]:
        raise ValueError("PTQ source weights differ from the scored checkpoint")
    witness = json.loads((folder/"software_parity_witness.json").read_text(encoding="utf-8"))
    expected_nodes = {n["id"] for n in json.loads((folder/"numeric_graph.json").read_text(encoding="utf-8"))["nodes"]} | {"image"}
    rows = witness.get("rows",[])
    if (witness.get("status") != "PASS_SOFTWARE_INTEGER_FAKE_PARITY" or witness.get("mismatches") != 0
            or len(rows) != len(expected_nodes) or {r["node"] for r in rows} != expected_nodes
            or any(r["mismatch_count"] != 0 or r["integer_sha256"] != r["fake_quantized_sha256"] for r in rows)):
        raise ValueError("Complete zero-mismatch software integer parity is required")
    actual = json.loads(Path(artifacts["quantization_result"]["path"]).read_text(encoding="utf-8"))
    fields = ("mAP50_95","checkpoint_sha256","quant_artifact_sha256","calibration_sha256","dataset_sha256")
    if (actual["arch_hash"] != record["arch_hash"] or actual["q_hash"] != record["q_hash"]
            or actual["seed"] != record["seed"] or actual["budget"] != record["budget"]
            or actual["split"] != "valid" or actual["test_evaluated"] is not False
            or any(actual[k] != record["accuracy"][k] for k in fields)):
        raise ValueError("Candidate accuracy differs from the actual quantization receipt")


def _point(record, config):
    from hw import evaluate, HardwareContext
    validate_record(record)
    if record.get("accuracy") is None:
        return None
    if not record.get("hardware_context"):
        raise ValueError("Scored candidates require hardware context")
    _verify_artifacts(record,require_hardware=record["hardware"]["status"]=="READY")
    if record["hardware"]["status"] != "READY":
        return None
    context = HardwareContext(**record["hardware_context"])
    # Re-read the original evidence; cached/copied READY flags are never trusted.
    actual = evaluate(Genotype.from_dict(record["genotype"]),
                      QuantizationPlan.from_dict(record["quantization"]), context,
                      evidence_path=record["artifacts"]["hardware_evidence"]["path"])
    hw = actual.to_dict()
    if hw != record["hardware"]:
        raise ValueError("Hardware receipt differs from revalidated evidence")
    scope = config["hardware"]["scope"]
    if hw["status"] != "READY" or hw["evidence_level"] != scope:
        return None
    if scope == "operator_hls" and config["hardware"].get("allow_operator_screening") is not True:
        return None
    if context.device != config["hardware"]["device"] or context.clock_mhz != config["hardware"]["clock_mhz"]:
        raise ValueError("Hardware platform/clock differs from experiment configuration")
    if context.backend != config["hardware"]["backend"] or context.tool_version != config["hardware"]["tool_version"]:
        raise ValueError("Hardware backend/tool version differs from experiment configuration")
    if context.mapping_policy not in config["mapping_policies"]:
        raise ValueError("Unknown mapping policy")
    point = dict(architecture_id=record["architecture_id"], arch_hash=record["arch_hash"],
                 profile=record["quantization"]["profile"], q_hash=record["q_hash"],
                 mapping_policy=context.mapping_policy, Accuracy=record["accuracy"]["mAP50_95"],
                 record_hash=record["record_hash"], evidence_level=scope,
                 checkpoint_sha256=record["accuracy"]["checkpoint_sha256"],
                 calibration_sha256=record["accuracy"]["calibration_sha256"],
                 dataset_sha256=record["accuracy"]["dataset_sha256"], deployment="NOT_CLAIMED")
    limits = config["hardware_budget"]["limits"]
    for old, name in METRICS.items():
        value = hw["metrics"].get(name)
        # HardwareResult metrics use plain standard-unit values.
        if isinstance(value, dict):
            value = value["value"]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            return None
        if value > limits[name]:
            return None
        point[old] = value
    point["cost_pressure"] = max(point[k]/limits[v] for k,v in METRICS.items())
    return point


def pareto_records(records, config):
    points, seen = [], set()
    protocols = set()
    for record in records:
        validate_record(record)
        protocols.add((record["git_commit"],record["config_hash"],record["source_manifest_hash"],
                       record["seed"],record["budget"],record["stage"]))
        if record["config_hash"] != canonical_hash(config):
            raise ValueError("Candidate configuration mismatch")
        key = (record["arch_hash"],record["q_hash"],record["hardware"]["context_hash"])
        if key in seen:
            raise ValueError("Duplicate candidate/quantization/mapping evidence")
        seen.add(key)
        point = _point(record, config)
        if point is not None:
            points.append(point)
    if len(protocols) > 1:
        raise ValueError("Do not pool different seeds/budgets/stages/source commits into one Pareto front")
    if len({p["dataset_sha256"] for p in points}) > 1:
        raise ValueError("Candidate dataset bindings differ")
    return policy.fronts(points)[0] if points else []


def select_stage(records, config, *, stage, previous):
    if type(stage) is not int or stage not in (0,1,2):
        raise ValueError("Stage must be 0, 1 or 2")
    if len(previous) != len(set(previous)):
        raise ValueError("Previous candidate hashes must be unique")
    expected = set(previous)
    seen, points, protocols, decisions = set(), [], set(), {}
    for record in records:
        validate_record(record)
        if record["seed"] != config["search_seed"] or record["budget"] != config["budgets"][stage] or record["stage"] != "search":
            raise ValueError("Stage records must match the registered seed/budget/search phase")
        if record["arch_hash"] not in expected or record["config_hash"] != canonical_hash(config):
            raise ValueError("Stage candidate/config mismatch")
        key = (record["arch_hash"],record["quantization"]["profile"],record["hardware"]["context_hash"])
        if key in seen:
            raise ValueError("Duplicate stage candidate")
        seen.add(key)
        mapping = (record.get("hardware_context") or {}).get("mapping_policy")
        pair = (record["quantization"]["profile"],mapping)
        if pair in decisions.setdefault(record["arch_hash"],set()):
            raise ValueError("Duplicate precision/mapping alternatives in one stage")
        decisions[record["arch_hash"]].add(pair)
        protocols.add((record["git_commit"],record["config_hash"],record["source_manifest_hash"]))
        point = _point(record, config)
        if point is not None:
            points.append(point)
    if len(protocols) > 1:
        raise ValueError("Stage records have inconsistent provenance")
    if {r["arch_hash"] for r in records} != expected:
        raise ValueError("Every previous candidate requires a recorded evaluation")
    required_pairs = {(q,m) for q in config["quant_profiles"] for m in config["mapping_policies"]}
    if any(pairs != required_pairs for pairs in decisions.values()):
        return dict(status="NEEDS_MEASUREMENT",selected=[],selected_hashes=[],ranked=[],boundary=[],points=points,
                    rejection_audits=[],reason="INCOMPLETE_PRECISION_MAPPING_DECISIONS",
                    requested_count=config["populations"][stage+1] if stage < 2 else config["shortlist"])
    if len({p["dataset_sha256"] for p in points}) > 1:
        raise ValueError("Candidate dataset bindings differ")
    target = config["populations"][stage+1] if stage < 2 else config["shortlist"]
    result = policy.promote(points, target, config["promotion"])
    result.update(status="READY" if len(result["selected"]) == target else "NEEDS_MEASUREMENT",
                  requested_count=target, budget=config["budgets"][stage], evidence_scope=config["hardware"]["scope"])
    by_id = {r["architecture_id"]:r["arch_hash"] for r in records}
    result["selected_hashes"] = [by_id[aid] for aid in result["selected"]]
    # Stable one-per-E0/E1 rejection audit, preserving the original campaign rule.
    result["rejection_audits"] = []
    if stage < 2:
        selected = set(result["selected"])
        for pan in (False,True):
            pool = sorted({r["architecture_id"] for r in records if r["genotype"]["pan"] == pan} - selected)
            if pool:
                chosen = min(pool, key=lambda aid: canonical_hash([config["search_seed"], stage, aid]))
                result["rejection_audits"].append(by_id[chosen])
    return result
