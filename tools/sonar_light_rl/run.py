"""Inspectable RL scheduling with explicit non-formal diagnostic/replay modes.

Formal run always calls the original new-space preflight. Its launch-driver and
evidence blockers are preserved. This module never invokes training, HLS or a
board while those gates remain open. Replay can verify software control flow,
not supply missing research measurements or three-seed confirmation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import yaml
from tools.modnas_comparison import joint_v2_elastic as base
from tools.sonar_joint_v2 import policy
from tools.sonar_light_rl.controller import JointController
from tools.sonar_light_rl.protocol import (cost_status, default_protocol, digest,
    environment_binding, joint_key, provisional_handoff, source_binding, validate_protocol, write_json)


def requirements():
    """Canonical order matches controller axes: architecture, profile, mapping."""
    from tools.modnas_elastic.model import architecture_hash
    return [dict(architecture_id=aid, architecture_hash=architecture_hash(arch),
        architecture=deepcopy(arch), num_classes=2, space_version="elastic_neck72_v1",
        profile=profile, mapping_policy=mapping)
        for aid, arch in sorted(base.architectures().items())
        for profile in policy.PROFILES for mapping in policy.POLICIES]


def make_plan(cfg):
    cfg = validate_protocol(cfg)
    base_path = (ROOT / cfg["base_config"]).resolve()
    if not base_path.is_relative_to(ROOT):
        raise ValueError("Base configuration must belong to this workspace")
    baseline = base.load_config(base_path)
    report = base.preflight(baseline)  # Never remove even apparently redundant blockers.
    blockers = list(report["blockers"])
    if not cfg["research_protocol_frozen"]:
        blockers.append("RL_REWARD_AND_EPISODE_RESEARCH_PROTOCOL_NOT_FROZEN")
    if (not isinstance(cfg["gpu_model"], str) or not cfg["gpu_model"].strip()
            or not policy.finite(cfg["gpu_h_cap"], positive=True)):
        blockers.append("RL_GPU_MODEL_AND_TOTAL_BUDGET_NOT_FROZEN")
    if cfg["hardware_budget"] != {k: baseline["hardware_budget"][k] for k in cfg["hardware_budget"]}:
        blockers.append("RL_AND_BASE_HARDWARE_BUDGET_DIFFER")
    if cfg["seed"] != baseline["search_seed"]:
        blockers.append("RL_AND_BASE_SEARCH_SEED_DIFFER")
    blockers.append("FORMAL_ACCOUNTED_EVALUATOR_AND_CONFIRMATION_DRIVER_NOT_CONNECTED")
    return dict(version=cfg["version"], status="BLOCKED_NOT_LAUNCHABLE",
        blockers=list(dict.fromkeys(blockers)), protocol=cfg, protocol_sha256=digest(cfg),
        execution_code_sha256=source_binding(), environment_binding=environment_binding(cfg),
        base_config_sha256=base.sha(base_path),
        base_preflight=report, formal_eligible=False, training_started=False,
        dataset_images_opened=False, test_accessed=False, launch_hardware=False,
        note="New RL strategy; does not replace or claim completion of enumerative V2")


class ReplayEvidence:
    """Offline records, always marked non-formal regardless of input labels."""
    def __init__(self, payload):
        if payload.get("schema_version") != "sonar_light_rl_replay_v1":
            raise ValueError("Explicit replay schema required")
        self.payload = deepcopy(payload)
        self.binding = digest(payload)
        self.costs, self.evaluations = {}, {}
        for cost in payload.get("costs", []):
            key = joint_key(cost)
            if key in self.costs:
                raise ValueError("Duplicate replay cost")
            self.costs[key] = cost
        for row in payload.get("evaluations", []):
            key = (row["architecture_id"], row["seed"], row["budget"])
            if key in self.evaluations:
                raise ValueError("Duplicate replay evaluation")
            if row.get("split") != "valid" or row.get("test_evaluated") is not False:
                raise ValueError("Replay must preserve validation-only selection")
            self.evaluations[key] = row

    def cost(self, requirement):
        return deepcopy(self.costs.get(joint_key(requirement), dict(requirement,
            status="NEEDS_MEASUREMENT", measurement_valid=False, timing_eligible=False,
            **{metric: None for metric in policy.METRICS})))

    def evaluate(self, aid, seed, budget):
        return deepcopy(self.evaluations.get((aid, seed, budget)))


def synthetic_evidence(cfg):
    """Deterministic control-flow fixture, with NO measured cost or accuracy claim."""
    costs, evaluations = [], []
    pool = requirements()
    for index, req in enumerate(pool):
        if index % 7 == 0:
            continue  # Exercise unknown evidence without inventing a penalty.
        values = {metric: limit * (.2 + (index % 11) / 20)
                  for metric, limit in cfg["hardware_budget"]["limits"].items()}
        if index % 5 == 0:
            values["Latency"] = cfg["hardware_budget"]["limits"]["Latency"] * 2
        costs.append(dict(req, **values, status="LOOKED_UP", measurement_valid=True,
            timing_eligible=True, evidence_kind="synthetic_diagnostic", formal_eligible=False))
    for index, aid in enumerate(sorted(base.architectures())):
        for budget in cfg["budgets"]:
            evaluations.append(dict(architecture_id=aid, seed=cfg["seed"], budget=budget,
                split="valid", test_evaluated=False, fp32=dict(mAP50_95=.1 + index / 1000),
                profiles={profile: dict(mAP50_95=.08 + index / 1000 + budget / 10000 + p / 10000,
                                        profile_sha256=digest([aid, profile, budget, "SYNTHETIC"]))
                          for p, profile in enumerate(policy.PROFILES)},
                checkpoint_sha256=digest([aid, budget, "SYNTHETIC_CHECKPOINT"]),
                calibration_sha256=digest("SYNTHETIC_CALIBRATION"),
                evidence_kind="synthetic_diagnostic", formal_eligible=False))
    return ReplayEvidence(dict(schema_version="sonar_light_rl_replay_v1", costs=costs,
                               evaluations=evaluations, evidence_kind="synthetic_diagnostic"))


def _fresh_state(cfg):
    return dict(stage=0, stage_attempts=0, total_attempts=0,
        eligible_architectures=sorted(base.architectures()), visited=[], events=[],
        measurement_queue={}, evaluation_queue=[], points_by_budget={},
        pareto_by_budget={}, completed_stages=[], status="RUNNING_DIAGNOSTIC")


def _save_session(output, state, controller, binding):
    payload = dict(binding=binding, state=state, controller=controller.state_dict())
    temporary = output / "state.tmp.pt"
    torch.save(payload, temporary)
    temporary.replace(output / "state.pt")
    # state.pt is the atomic authority; JSON views may be rebuilt after a crash.
    write_json(output / "events.json", dict(formal_eligible=False, events=state["events"]))
    write_json(output / "measurement_queue.json", dict(formal_eligible=False,
        tasks=list(state["measurement_queue"].values())))


def run_replay(cfg, evidence, output, *, resume=False, max_attempts=None):
    """Execute no-duplicate masked sampling, feedback and stage-isolated Pareto.

    max_attempts is a diagnostic interruption boundary, not a research budget.
    Missing measurement/evaluation never calls controller.update. All output is
    explicitly non-formal and unconfirmed, including when input is real replay.
    """
    cfg = validate_protocol(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = dict(protocol_sha256=digest(cfg), evidence_sha256=evidence.binding,
                   execution_code_sha256=source_binding(), environment_binding=environment_binding(cfg))
    controller = JointController(cfg)
    state = _fresh_state(cfg)
    if resume:
        saved = torch.load(output / "state.pt", map_location="cpu", weights_only=True)
        if saved["binding"] != binding:
            raise ValueError("Resume protocol/evidence/source drift")
        state = saved["state"]
        controller.load_state_dict(saved["controller"])
    elif any(output.iterdir()):
        raise FileExistsError("Choose empty output or explicit resume; preserve existing evidence")
    pool = requirements()
    attempts_this_call = 0
    while state["stage"] < len(cfg["budgets"]):
        stage = state["stage"]
        budget = cfg["budgets"][stage]
        bucket = str(budget)
        allowed = set(state["eligible_architectures"])
        visited = set(state["visited"])
        available = [r["architecture_id"] in allowed and i not in visited for i, r in enumerate(pool)]
        if state["stage_attempts"] >= cfg["episodes_per_stage"][stage] or not any(available):
            points = state["points_by_budget"].get(bucket, [])
            state["completed_stages"].append(budget)
            if stage < 2 and points:
                promoted = policy.promote(points, cfg["populations"][stage + 1], cfg["promotion"])
                state["eligible_architectures"] = promoted["selected"]
            elif not points:
                state["status"] = "BLOCKED_NO_EVALUATED_FEASIBLE_POINTS"
                break
            state.update(stage=stage + 1, stage_attempts=0, visited=[])
            continue
        if max_attempts is not None and attempts_this_call >= max_attempts:
            state["status"] = "INTERRUPTED_DIAGNOSTIC"
            break
        index, log_probability, entropy = controller.sample(available)
        req = pool[index]
        cost = evidence.cost(req)
        status = cost_status(cost, req, cfg)
        event = dict(episode=state["total_attempts"], budget=budget, candidate=deepcopy(req),
            status=status, reward=None, controller_updated=False,
            log_probability=float(log_probability.detach()), formal_eligible=False,
            evidence_kind="synthetic_diagnostic" if evidence.payload.get("evidence_kind") == "synthetic_diagnostic" else "offline_replay")
        if status == "NEEDS_MEASUREMENT":
            state["measurement_queue"]["|".join(joint_key(req))] = dict(req,
                status="NEEDS_MEASUREMENT", origin=event["evidence_kind"], formal_eligible=False)
        elif status == "OVER_BUDGET":
            reward = controller.compute_reward(cost, feasible=False)
            event.update(reward=reward, loss=controller.update(log_probability, entropy, reward), controller_updated=True)
        else:
            evaluation = evidence.evaluate(req["architecture_id"], cfg["seed"], budget)
            if evaluation is None:
                event["status"] = "NEEDS_EVALUATION"
                state["evaluation_queue"].append(dict(req, seed=cfg["seed"], budget=budget))
            else:
                point = policy.joint_points([evaluation], [cost], cfg["hardware_budget"])[0]
                point.update(evidence_kind=event["evidence_kind"], formal_eligible=False)
                points = state["points_by_budget"].setdefault(bucket, [])
                points.append(point)
                state["pareto_by_budget"][bucket] = policy.fronts(points)[0]
                reward = controller.compute_reward(cost, point["Accuracy"], feasible=True)
                event.update(reward=reward, loss=controller.update(log_probability, entropy, reward), controller_updated=True)
        state["events"].append(event)
        state["visited"].append(index)
        state["stage_attempts"] += 1
        state["total_attempts"] += 1
        attempts_this_call += 1
        _save_session(output, state, controller, binding)
    if state["stage"] == 3:
        state["status"] = "COMPLETE_DIAGNOSTIC_NOT_RESEARCH"
    _save_session(output, state, controller, binding)
    summary = dict(status=state["status"], binding=binding, protocol=cfg,
        total_attempts=state["total_attempts"], controller_updates=controller.updates,
        pareto_by_budget=state["pareto_by_budget"], formal_eligible=False,
        completed_stages=state["completed_stages"], training_started=False,
        launch_hardware=False, test_accessed=False, confirmation="NOT_EXECUTED",
        pending_handoffs=[provisional_handoff(p) for p in state["pareto_by_budget"].get("80", [])])
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "diagnostic", "replay", "run"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = validate_protocol(yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
                            if args.protocol else default_protocol())
    if args.command in ("plan", "run"):
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Use a new output directory")
        report = make_plan(cfg)
        write_json(args.output / "plan.json", report)
        print(json.dumps(dict(status=report["status"], blockers=report["blockers"])))
        return 2 if args.command == "run" else 0
    if args.command == "replay" and args.evidence is None:
        parser.error("replay requires --evidence")
    evidence = (synthetic_evidence(cfg) if args.command == "diagnostic" else
                ReplayEvidence(json.loads(args.evidence.read_text(encoding="utf-8"))))
    report = run_replay(cfg, evidence, args.output, resume=args.resume)
    print(json.dumps({k: report[k] for k in ("status", "total_attempts", "controller_updates", "formal_eligible")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
