"""Fail-closed formal-readiness helpers for the benchmark campaign."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Mapping


def make_requirement(
    requirement_id: str,
    description: str,
    *,
    passed: bool,
    observed: Any,
    required: Any,
    evidence: Iterable[Mapping[str, Any]] = (),
    blockers: Iterable[str] = (),
) -> dict[str, Any]:
    blocker_list = [str(value) for value in blockers if str(value)]
    if not passed and not blocker_list:
        blocker_list = ["required evidence is incomplete or missing"]
    return {
        "requirement_id": str(requirement_id),
        "description": str(description),
        "status": "PASS" if passed else "PENDING",
        "passed": bool(passed),
        "observed": observed,
        "required": required,
        "evidence": [dict(value) for value in evidence],
        "blockers": blocker_list,
    }


def build_readiness_report(
    campaign_id: str, requirements: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [dict(row) for row in requirements]
    ids = [str(row.get("requirement_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("formal-readiness requirement ids must be unique")
    all_pass = bool(rows) and all(row.get("passed") is True for row in rows)
    return {
        "schema_version": 1,
        "campaign_id": str(campaign_id),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "status": "READY" if all_pass else "NOT_READY",
        "formal_execution_ready": all_pass,
        "passed_count": sum(row.get("passed") is True for row in rows),
        "requirement_count": len(rows),
        "requirements": rows,
        "blocking_requirement_ids": [
            str(row.get("requirement_id"))
            for row in rows
            if row.get("passed") is not True
        ],
        "boundary": (
            "READY requires every named evidence layer. Smoke, source presence, author-"
            "reported numbers, or cross-platform artifacts never satisfy a formal result."
        ),
    }
