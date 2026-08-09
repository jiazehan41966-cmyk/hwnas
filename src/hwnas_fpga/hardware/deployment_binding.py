"""Fail-closed ZCU104 deployment artifact binding helpers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(SHA256_PATTERN.fullmatch(str(value or "")))


def audit_zcu104_deployment_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Audit identity and artifact bindings without promoting deployment gates."""
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else {}
    platform = payload.get("platform") if isinstance(payload.get("platform"), Mapping) else {}
    transport = payload.get("transport") if isinstance(payload.get("transport"), Mapping) else {}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), Mapping) else {}
    toolchain = payload.get("toolchain") if isinstance(payload.get("toolchain"), Mapping) else {}

    required_artifacts = ("checkpoint", "bitstream", "xsa", "hwh", "runtime")
    artifact_gates: dict[str, bool] = {}
    for name in required_artifacts:
        record = artifacts.get(name) if isinstance(artifacts.get(name), Mapping) else {}
        artifact_gates[name] = bool(str(record.get("path") or "").strip()) and _valid_sha256(
            record.get("sha256")
        )

    gates = {
        "campaign_id": str(payload.get("campaign_id") or "") == "zcu104_full_campaign_v1",
        "candidate_id": bool(str(candidate.get("arch_id") or "").strip()),
        "architecture_sha256": _valid_sha256(candidate.get("architecture_sha256")),
        "board_model": str(platform.get("board_model") or "").upper() == "ZCU104",
        "target_part": str(platform.get("target_part") or "")
        == "xczu7ev-ffvc1156-2-e",
        "board_serial": bool(str(platform.get("board_serial") or "").strip()),
        "transport_type": str(transport.get("type") or "") == "axi_dma",
        "transport_id": bool(str(transport.get("id") or "").strip()),
        "vivado_2022_2": str(toolchain.get("vivado_version") or "") == "2022.2",
        "vitis_hls_2022_2": str(toolchain.get("vitis_hls_version") or "") == "2022.2",
        "toolchain_sha256": _valid_sha256(toolchain.get("sha256")),
        **{f"artifact_{name}": passed for name, passed in artifact_gates.items()},
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "overall_pass": all(gates.values()),
        "gates": gates,
        "claim_boundary": (
            "PASS proves artifact identity binding only. It does not prove route timing, "
            "board classification, latency, zero mismatch, or measured power."
        ),
    }
