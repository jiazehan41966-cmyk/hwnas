"""Load and pin FPGA H2 NUMERIC_CONTRACT_V1. Do not invent a second arithmetic rule."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_JSON = ROOT / "reports/fpga_h2_contract/numeric_contract_v1.json"
EXPECTED_HASH = "001f461391eeaebd05daa984ff0703f76cc3ddd71cd347b203bf3e4c1ba7d145"


def contract_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def contract_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(contract_bytes(payload)).hexdigest()


def load_numeric_contract_v1() -> dict[str, Any]:
    payload = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    digest = contract_hash(payload)
    if digest != EXPECTED_HASH:
        raise ValueError(
            "NUMERIC_CONTRACT_V1 hash mismatch; Phase 5A must not create a new contract"
        )
    return payload


def require_h2_aligned(config: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> None:
    """Reject any software policy that would silently replace H2 arithmetic."""
    spec = contract or load_numeric_contract_v1()
    if int(config.get("activation_bits", spec["activation_bits"])) != spec["activation_bits"]:
        raise ValueError("activation bitwidth is not H2 INT8")
    if int(config.get("weight_bits", spec["weight_bits"])) != spec["weight_bits"]:
        raise ValueError("weight bitwidth is not H2 INT8")
    if int(config.get("accumulator_bits", spec["accumulator_bits"])) != spec["accumulator_bits"]:
        raise ValueError("accumulator bitwidth is not H2 INT32")
    if list(config.get("signed_range", spec["range"])) != list(spec["range"]):
        raise ValueError("saturation range is not H2 [-128,127]")
    if int(config.get("zero_point", spec["zero_point"])) != spec["zero_point"]:
        raise ValueError("H2 zero_point is fixed at 0; asymmetric FPGA policy is forbidden")
    if str(config.get("rounding", spec["rounding"])) != spec["rounding"]:
        raise ValueError("rounding is not H2 nearest_even")
    if int(config.get("requant_product_bits", spec["requant_product_bits"])) != spec["requant_product_bits"]:
        raise ValueError("requant product width is not H2 INT64")
    if str(config.get("scale", spec["scale"])) != spec["scale"]:
        raise ValueError("scale representation is not H2 power-of-two")
