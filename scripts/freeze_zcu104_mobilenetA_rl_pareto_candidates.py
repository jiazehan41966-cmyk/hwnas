"""Freeze selected ZCU104 MobileNetA RL Pareto candidates for retraining."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.hardware.mobilenetA_operator_cost import load_hashed_json
from hwnas_fpga.interfaces import SearchCandidate, CandidateMetrics
from hwnas_fpga.search.pareto import build_pareto_representative_roles

from hwnas_fpga.fourstage_operator import architecture_sha256
from hwnas_fpga.search_space import ArchitectureSpec


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "zcu104_full_campaign_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def candidate_metric(candidate: dict[str, Any], name: str) -> float | None:
    metrics = candidate.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def canonical_candidate_record(
    *,
    selected: dict[str, Any],
    source_candidate_path: Path,
    search_config_path: Path,
    pareto_selection_path: Path,
    lut_path: Path,
    classification_setup_summary_path: Path | None,
    role_names: list[str],
) -> dict[str, Any]:
    bound_config, bound_lut = validate_run_binding(pareto_selection_path, search_config_path, lut_path)
    architecture = ArchitectureSpec.from_dict(selected["encoding"])
    arch_hash = architecture_sha256(architecture)
    source_payload = read_json(source_candidate_path)
    source_candidate = source_payload.get("candidate") or {}
    if source_candidate.get("arch_id") != selected["arch_id"]:
        raise ValueError("source candidate identity mismatch")
    expected_source = pareto_selection_path.parent / "candidates" / (selected["arch_id"] + ".json")
    if source_candidate_path.resolve() != expected_source.resolve():
        raise ValueError("candidate is not from the selected search run")
    eligible, _roles = recorded_selection(pareto_selection_path.parent, read_json(pareto_selection_path))
    if not any(row["arch_id"] == selected["arch_id"] and row["encoding"] == selected["encoding"] for row in eligible):
        raise ValueError("candidate was not selected by this run")
    for axis in read_json(pareto_selection_path).get("objectives", []):
        if source_candidate.get("metrics", {}).get(axis) != selected.get("metrics", {}).get(axis):
            raise ValueError("candidate metric mismatch: " + axis)
    source_arch = source_candidate.get("encoding")
    if source_arch != selected["encoding"]:
        raise RuntimeError(f"candidate encoding mismatch: {source_candidate_path}")

    payload = {
        "schema_version": 1,
        "artifact_type": "frozen_zcu104_mobilenetA_rl_pareto_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "method": "rl",
        "search_space_version": bound_config["search_space"]["search_space_version"],
        "candidate": {
            "arch_id": selected["arch_id"],
            "architecture_sha256": arch_hash,
            "encoding": selected["encoding"],
            "metrics": selected.get("metrics") or {},
            "objective_values": selected.get("objective_values") or {},
            "feasible": bool(selected.get("feasible", True)),
            "violations": selected.get("violations") or [],
        },
        "selection": {
            "roles": role_names,
            "selection_index": selected.get("selection_index"),
            "source_rank": selected.get("rank"),
            "source_candidate_index": selected.get("candidate_index"),
            "rule": "recorded selected_topk plus recorded representatives from the full Pareto front",
        },
        "search_config": {
            "path": rel(search_config_path),
            "sha256": sha256_file(search_config_path),
        },
        "source_search_artifacts": {
            "pareto_representatives": {
                "path": rel(pareto_selection_path.parent / "pareto_representatives.json"),
                "sha256": sha256_file(pareto_selection_path.parent / "pareto_representatives.json"),
            },
            "lut_stats": {
                "path": rel(pareto_selection_path.parent / "lut_stats.json"),
                "sha256": sha256_file(pareto_selection_path.parent / "lut_stats.json"),
            },
            "pareto_selection": {
                "path": rel(pareto_selection_path),
                "sha256": sha256_file(pareto_selection_path),
            },
            "candidate_json": {
                "path": rel(source_candidate_path),
                "sha256": sha256_file(source_candidate_path),
            },
        },
        "lut_binding": {
            "path": rel(lut_path),
            "sha256": sha256_file(lut_path),
            "version": bound_lut["hardware_lut_version"],
            "cost_semantics_version": read_json(pareto_selection_path.parent / "lut_stats.json").get(
                "cost_semantics_version", "historical_unversioned"
            ),
            "power_boundary": "power_w is a vectorless operator-cost proxy, not measured board power",
        },
        "classification_setup_summary": None,
        "claim_boundary": (
            "Frozen search candidate only. Final classification, INT8, full route, board "
            "accuracy, and measured power require separate downstream evidence."
        ),
    }
    if classification_setup_summary_path is not None:
        payload["classification_setup_summary"] = {
            "path": rel(classification_setup_summary_path),
            "sha256": sha256_file(classification_setup_summary_path),
        }
    return payload


def pick_roles(selected: list[dict[str, Any]]) -> dict[str, str]:
    """Compatibility helper; one shared definition of representative roles."""
    axes = ["macro_f1", "latency_ms", "energy_mj", "dsp", "bram", "lut"]
    candidates = [SearchCandidate(arch_id=row["arch_id"], encoding=row.get("encoding", {}),
        metrics=CandidateMetrics(**{k:v for k,v in row["metrics"].items()
                                    if k in CandidateMetrics.__dataclass_fields__})) for row in selected]
    result = build_pareto_representative_roles(candidates, objectives=axes,
                                              directions=["max"] + ["min"]*5)
    return {name: row["arch_id"] for name,row in result["roles"].items()
            if row and name in {"accuracy_first", "knee_point", "latency_energy_first"}}


def validate_run_binding(pareto_selection_path: Path, search_config_path: Path, lut_path: Path):
    results_dir = pareto_selection_path.resolve().parent
    saved_config_path = results_dir.parent / "config.yaml"
    saved = yaml.safe_load(saved_config_path.read_text(encoding="utf-8"))
    supplied = yaml.safe_load(search_config_path.read_text(encoding="utf-8"))
    if saved != supplied:
        raise ValueError("freeze config differs from the saved search run config")
    if saved.get("search", {}).get("method") != "rl":
        raise ValueError("freeze requires an RL search run")
    version = saved.get("search_space", {}).get("search_space_version")
    if version != "mobilenetA_full_dmdc_cfem_v1":
        raise ValueError("freeze search-space version is not MobileNetA")
    stats = read_json(results_dir / "lut_stats.json")
    _, lut = load_hashed_json(lut_path)
    if not lut.get("rows") or lut.get("search_ready") is not True:
        raise ValueError("freeze LUT must contain ready operator evidence")
    if lut.get("search_space_version") != version:
        raise ValueError("freeze LUT/search space mismatch")
    if stats.get("operator_cost_file_sha256") != sha256_file(lut_path):
        raise ValueError("freeze LUT differs from the LUT consumed by this run")
    if stats.get("payload_sha256") != lut.get("payload_sha256"):
        raise ValueError("freeze LUT payload differs from recorded run binding")
    if stats.get("hardware_lut_version") != lut.get("hardware_lut_version"):
        raise ValueError("freeze LUT version differs from recorded run binding")
    configured_path = Path(saved["hardware"]["operator_cost_lut_path"])
    configured_path = configured_path if configured_path.is_absolute() else ROOT/configured_path
    if configured_path.is_file() and sha256_file(configured_path) != sha256_file(lut_path):
        raise ValueError("freeze LUT differs from the saved configuration's file")
    return saved, lut


def recorded_selection(results_dir: Path, pareto: dict):
    """Preserve original roles, including representatives outside selected Top-K."""
    representatives = read_json(results_dir / "pareto_representatives.json")
    if representatives.get("status") != "complete":
        raise ValueError("recorded Pareto representatives are incomplete")
    for key in ("objectives", "directions"):
        if representatives.get(key) != pareto.get(key):
            raise ValueError("representative objective protocol differs from Pareto selection")
    selected = list(pareto.get("selected_topk") or [])
    rows = {row["arch_id"]: row for row in pareto.get("ranked_candidates", [])}
    objective_names = tuple(pareto.get("objectives") or ())
    for record in selected:
        source = rows.get(record["arch_id"])
        if source is None or source["encoding"] != record["encoding"]:
            raise ValueError("selected Top-K candidate differs from ranked candidates")
        if any(
            source.get("metrics", {}).get(axis) != record.get("metrics", {}).get(axis)
            for axis in objective_names
        ):
            raise ValueError("selected Top-K candidate objective metrics differ from ranked candidates")
    roles = {}
    for name, record in representatives.get("roles", {}).items():
        if record is None:
            continue
        source = rows.get(record["arch_id"])
        if source is None or not source.get("pareto_front") or source["encoding"] != record["encoding"]:
            raise ValueError("representative identity is not on the recorded Pareto front")
        if any(source.get("metrics", {}).get(axis) != record.get("metrics", {}).get(axis)
               for axis in pareto["objectives"]):
            raise ValueError("representative metrics differ from the recorded Pareto front")
        roles[name] = record["arch_id"]
        if not any(row["arch_id"] == record["arch_id"] for row in selected):
            selected.append(source)
    return selected, roles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--lut-path",
        default=None,
    )
    parser.add_argument(
        "--classification-setup-summary",
        default="artifacts/zcu104_full_campaign_v1/classification_setup/classification_setup_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    results_dir = run_dir / "results"
    pareto_selection_path = results_dir / "pareto_selection.json"
    if not pareto_selection_path.is_file():
        raise FileNotFoundError(pareto_selection_path)
    pareto = read_json(pareto_selection_path)
    selected = pareto.get("selected_topk") or []
    if not selected:
        raise RuntimeError("pareto_selection.json has no selected_topk candidates")

    search_config_path = (ROOT / args.config).resolve() if args.config else run_dir / "config.yaml"
    saved_config = yaml.safe_load((run_dir/"config.yaml").read_text(encoding="utf-8"))
    lut_path = (ROOT / (args.lut_path or saved_config["hardware"]["operator_cost_lut_path"])).resolve()
    bound_config, bound_lut = validate_run_binding(pareto_selection_path, search_config_path, lut_path)
    setup_path = (ROOT / args.classification_setup_summary).resolve()
    setup_path = setup_path if setup_path.is_file() else None
    output_dir = (ROOT / args.output_dir).resolve() if args.output_dir else run_dir/"frozen_candidates_v2"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("freeze output must be a new or empty directory")
    candidate_output_dir = output_dir / "candidates"
    selected, roles_by_name = recorded_selection(results_dir, pareto)

    frozen_candidates = []
    pending_writes = []
    roles: dict[str, Any] = {}
    for topk_index, item in enumerate(selected, start=1):
        arch_id = str(item["arch_id"])
        role_names = [role for role, role_arch_id in roles_by_name.items() if role_arch_id == arch_id]
        if not role_names:
            role_names = [f"pareto_top{topk_index}"]
        source_candidate_path = results_dir / "candidates" / f"{arch_id}.json"
        payload = canonical_candidate_record(
            selected=item,
            source_candidate_path=source_candidate_path,
            search_config_path=search_config_path,
            pareto_selection_path=pareto_selection_path,
            lut_path=lut_path,
            classification_setup_summary_path=setup_path,
            role_names=role_names,
        )
        out_path = candidate_output_dir / f"{arch_id}.frozen_candidate.json"
        pending_writes.append((out_path, payload))
        file_sha = hashlib.sha256(
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        ).hexdigest()
        candidate_summary = {
            "arch_id": arch_id,
            "architecture_sha256": payload["candidate"]["architecture_sha256"],
            "roles": role_names,
            "candidate_json": {"path": rel(out_path), "sha256": file_sha},
            "source_candidate_json": payload["source_search_artifacts"]["candidate_json"],
            "macro_f1": candidate_metric(item, "macro_f1"),
            "top1": candidate_metric(item, "top1"),
            "latency_ms": candidate_metric(item, "latency_ms"),
            "energy_mj": candidate_metric(item, "energy_mj"),
            "lut": candidate_metric(item, "lut"),
            "dsp": candidate_metric(item, "dsp"),
            "bram": candidate_metric(item, "bram"),
            "power_w_proxy": candidate_metric(item, "power_w"),
        }
        frozen_candidates.append(candidate_summary)
        for role in role_names:
            roles[role] = candidate_summary

    manifest = {
        "schema_version": 1,
        "artifact_type": "zcu104_mobilenetA_rl_pareto_candidate_freeze",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "method": "rl",
        "search_run": {"path": rel(run_dir)},
        "search_protocol": {
            "config": rel(search_config_path),
            "config_sha256": sha256_file(search_config_path),
            "fold": bound_config["dataset"]["fold"],
            "seed": bound_config["project"]["seed"],
            "split_mode": bound_config["dataset"]["split_mode"],
            "candidate_evaluations": bound_config["search"]["episodes"],
            "eval_epochs": bound_config["search"]["eval_epochs"],
            "eval_early_stopping_patience": bound_config["search"].get("eval_early_stopping_patience"),
            "candidate_records": int(pareto.get("candidate_count", 0)),
            "pareto_front_size": int(pareto.get("pareto_front_size", 0)),
            "hypervolume": pareto.get("hypervolume"),
            "objectives": pareto.get("objectives"),
            "directions": pareto.get("directions"),
        },
        "lut_binding": {
            "path": rel(lut_path),
            "sha256": sha256_file(lut_path),
            "version": bound_lut["hardware_lut_version"],
            "power_boundary": "power_w is a vectorless operator-cost proxy, not measured board power",
        },
        "source_search_artifacts": {
            "pareto_selection": {
                "path": rel(pareto_selection_path),
                "sha256": sha256_file(pareto_selection_path),
            },
            "rl_search_metrics": {
                "path": rel(results_dir / "rl_search_metrics.json"),
                "sha256": sha256_file(results_dir / "rl_search_metrics.json"),
            },
            "rl_search_curve": {
                "path": rel(results_dir / "rl_search_curve.csv"),
                "sha256": sha256_file(results_dir / "rl_search_curve.csv"),
            },
        },
        "role_selection_source": {"path": rel(results_dir / "pareto_representatives.json"), "sha256": sha256_file(results_dir / "pareto_representatives.json")},
        "role_selection_rules": {
            "accuracy_first": "preserve recorded accuracy_first from full Pareto front",
            "knee_point": "preserve recorded knee_point from full Pareto front",
            "latency_energy_first": "preserve recorded latency_energy_first (latency before energy)",
            "pareto_topN": "remaining selected_topk candidates in recorded order",
        },
        "roles": roles,
        "frozen_candidates": frozen_candidates,
        "unique_retrain_candidates": frozen_candidates,
        "roles_may_overlap": True,
        "claim_boundary": (
            "This is a post-search freeze of selected RL Pareto candidates. It does not "
            "claim final test performance, INT8 parity, full-route timing, board accuracy, "
            "or measured board power."
        ),
    }
    manifest_path = output_dir / "rl_frozen_mobilenetA_pareto_candidates.json"
    # Validate every candidate and all manifest sources before publishing any files.
    for out_path, payload in pending_writes:
        write_json(out_path, payload)
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": rel(manifest_path), "sha256": sha256_file(manifest_path), "candidate_count": len(frozen_candidates), "roles": roles_by_name}, indent=2))


if __name__ == "__main__":
    main()
