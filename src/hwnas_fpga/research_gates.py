"""Repository-level gates that prevent evidence classes from being over-claimed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY = 15_728_640
STAGE3_ALLOWED_METHODS = frozenset(
    {
        "aging_evolution",
        "enumeration",
        "hierarchical_random",
        "proxyless",
        "random",
        "rl",
        "spos",
    }
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def stage3_gate_status(
    repo_root: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the frozen-stage-3 decision without starting any search work."""

    root = Path(repo_root).resolve()
    cfg = dict(config or {})
    gate_cfg = cfg.get("stage3_gate", {})
    if not isinstance(gate_cfg, Mapping):
        raise ValueError("stage3_gate must be a mapping")

    calibration_path = root / str(
        gate_cfg.get(
            "calibration_v2",
            "artifacts/hw_surrogate_calibration_v2/calibration_v2.json",
        )
    )
    board_path = root / str(
        gate_cfg.get(
            "board_validation",
            "results/g4_rl_arch_193_fold1_seed42/board_validation_summary.json",
        )
    )
    approval_path = root / str(
        gate_cfg.get(
            "replan_approval",
            "artifacts/stage3_replan_approval.json",
        )
    )
    gate0_path = root / str(
        gate_cfg.get(
            "proxy_reliability_gate0",
            "artifacts/proxy_reliability_gate0/manifest_summary_v2.json",
        )
    )
    g5_path = root / str(
        gate_cfg.get(
            "sonar_operator_gate",
            "artifacts/sonar_operator_gate/sonar_operator_gate.json",
        )
    )
    calibration = _read_json(calibration_path)
    board = _read_json(board_path)
    approval = _read_json(approval_path)
    gate0 = _read_json(gate0_path)
    g5 = _read_json(g5_path)
    gate0_work_units = int((gate0 or {}).get("work_unit_count", 0) or 0)
    gate0_completed = int((gate0 or {}).get("formal_completed_work_units", 0) or 0)
    gate0_status = str((gate0 or {}).get("gate_status", "")).strip().lower()
    gate0_pass = bool(
        gate0
        and gate0_status in {"pass", "passed", "ready"}
        and gate0_work_units > 0
        and gate0_completed >= gate0_work_units
    )
    selected_methods: list[str] = []
    if approval:
        if isinstance(approval.get("methods"), list):
            selected_methods = [str(value) for value in approval["methods"]]
        elif approval.get("method") is not None:
            selected_methods = [str(approval["method"])]
    methods_are_valid = bool(selected_methods) and all(
        method in STAGE3_ALLOWED_METHODS for method in selected_methods
    )
    search_cfg = cfg.get("search", {})
    requested_method = (
        str(search_cfg.get("method"))
        if isinstance(search_cfg, Mapping) and search_cfg.get("method") is not None
        else None
    )
    if requested_method == "aging":
        requested_method = "aging_evolution"
    search_space_cfg = cfg.get("search_space", {})
    configured_ops = {
        str(value)
        for value in (
            search_space_cfg.get("op_choices", [])
            if isinstance(search_space_cfg, Mapping)
            else []
        )
    }
    sonar_ops = sorted(configured_ops & {"denoise", "edge"})
    g5_required = bool(sonar_ops)
    g5_pass = bool(not g5_required or (g5 and g5.get("overall_pass") is True))
    gates = {
        "gate0_proxy_reliability_pass": gate0_pass,
        "g2_pass": bool(calibration and calibration.get("g2_pass") is True),
        "g4_claimable": bool(board and board.get("claimable") is True),
        "g4_zero_numeric_mismatch": bool(
            board and int(board.get("numeric_mismatch_count", -1)) == 0
        ),
        "stage3_replan_approved": bool(
            approval and approval.get("approved") is True
        ),
        "stage3_method_selected": bool(
            methods_are_valid
        ),
        "stage3_requested_method_approved": bool(
            methods_are_valid
            and (
                requested_method is None
                or requested_method in selected_methods
            )
        ),
        "g5_sonar_operator_pass_or_not_required": g5_pass,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "stage": 3,
        "status": "PASS" if passed else "FROZEN",
        "may_start_new_claimable_search": passed,
        "semantic_safe_search_space_cardinality": (
            SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY
        ),
        "gates": gates,
        "requested_method": requested_method,
        "approved_methods": selected_methods,
        "allowed_methods": sorted(STAGE3_ALLOWED_METHODS),
        "gate0_progress": {
            "gate_status": gate0_status or None,
            "formal_completed_work_units": gate0_completed,
            "work_unit_count": gate0_work_units,
        },
        "g5_required": g5_required,
        "g5_required_operators": sonar_ops,
        "evidence": {
            "calibration_v2": str(calibration_path),
            "board_validation": str(board_path),
            "replan_approval": str(approval_path),
            "proxy_reliability_gate0": str(gate0_path),
            "sonar_operator_gate": str(g5_path),
        },
        "boundary": (
            "Legacy searches remain exploratory evidence. A new claimable "
            "NKSID search requires Gate0, G2, G4, any required G5 sonar-operator "
            "admission, an explicit post-G4 stage-3 approval, and the requested "
            "method must appear in that approval."
        ),
    }


def require_stage3_search_gate(
    repo_root: str | Path,
    *,
    dataset_name: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Permit dummy smoke runs, but block a new real-data search while frozen."""

    status = stage3_gate_status(repo_root, config=config)
    if dataset_name == "dummy":
        return {
            **status,
            "status": "SMOKE_ONLY",
            "may_start_new_claimable_search": False,
            "dummy_smoke_permitted": True,
        }
    if not status["may_start_new_claimable_search"]:
        failed = [name for name, passed in status["gates"].items() if not passed]
        raise RuntimeError(
            "Stage 3 is frozen; refusing to start a real-data search. "
            f"Failed gates: {', '.join(failed)}. "
            f"Search-space cardinality is "
            f"{SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY:,}."
        )
    return status
