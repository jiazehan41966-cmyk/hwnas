"""Repository-level gates that prevent evidence classes from being over-claimed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SEMANTIC_SAFE_SEARCH_SPACE_CARDINALITY = 15_728_640


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
    calibration = _read_json(calibration_path)
    board = _read_json(board_path)
    approval = _read_json(approval_path)
    gates = {
        "g2_pass": bool(calibration and calibration.get("g2_pass") is True),
        "g4_claimable": bool(board and board.get("claimable") is True),
        "g4_zero_numeric_mismatch": bool(
            board and int(board.get("numeric_mismatch_count", -1)) == 0
        ),
        "stage3_replan_approved": bool(
            approval and approval.get("approved") is True
        ),
        "stage3_method_selected": bool(
            approval
            and approval.get("method")
            in {"enumeration", "spos", "hierarchical_random"}
        ),
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
        "evidence": {
            "calibration_v2": str(calibration_path),
            "board_validation": str(board_path),
            "replan_approval": str(approval_path),
        },
        "boundary": (
            "RL/Random/Proxyless remain legacy exploratory evidence. A new "
            "claimable NKSID search requires G2, G4, and an explicit post-G4 "
            "stage-3 method decision."
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
