"""Risk checks: identical trunk IDs must never alias heads, raw/EMA or mappings."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.sonar_192_chain import contract as c
from tools.sonar_192_chain.model import inspect_checkpoint


def sample_identity():
    spec = c.model_contract("9ee366")
    cp = dict(sha256="a"*64, model_state_sha256="b"*64, weights="ema", validation_weights="ema",
              epoch=1, seed=42, mode="smoke", formal_eligible=False, selection_policy="epoch80_ema",
              full_contract_sha256=spec["full_contract_sha256"])
    mapping = dict(implementation_id="TEST_ONLY", PE=1, SIMD=1, buffer_profile="TEST_ONLY",
                   numeric_contract_sha256="c"*64, interface_contract_sha256="d"*64, status="PROPOSED")
    calibration = dict(sha256="e"*64, split="train", test_accessed=False,
                       number_of_calibration_images=1, source_verified=False)
    return c.make_identity("9ee366", checkpoint=cp, profile="W8A8", mapping=mapping, calibration=calibration)


def resign(value):
    value.pop("identity_sha256", None)
    value["identity_sha256"] = c.digest(value)
    return value


def test_frozen_contract_and_structure_are_distinct():
    spec = c.model_contract("9ee366")
    assert spec["space_version"] == "sonar_backbone_neck192_v2"
    assert spec["head_contract"]["raw_channels_per_level"] == 6
    assert spec["architecture_sha256"] != spec["full_contract_sha256"]
    assert c.model_contract(spec["architecture"]) == spec


@pytest.mark.parametrize("field,value", [("space_version", "elastic_neck72_v1"),
    ("full_contract_sha256", "0"*64), ("head_contract_sha256", "f"*64)])
def test_resigned_wrong_head_or_space_still_rejected(field, value):
    identity = sample_identity()
    identity[field] = value
    with pytest.raises(ValueError, match="model/head/source"):
        c.validate_identity(resign(identity))


def test_smoke_cannot_be_promoted_with_a_flag():
    identity = sample_identity()
    identity["purpose"] = "formal"
    identity["mapping"]["status"] = "FROZEN_CONCRETE_MAPPING"
    identity["checkpoint"]["formal_eligible"] = True
    with pytest.raises(ValueError, match="epoch80"):
        c.validate_identity(resign(identity))


def test_raw_weights_rejected_and_mapping_changes_identity():
    first = sample_identity()
    second = deepcopy(first)
    second["checkpoint"]["weights"] = "raw"
    with pytest.raises(ValueError, match="EMA"):
        c.validate_identity(resign(second))
    second = deepcopy(first)
    second["mapping"]["PE"] = 2
    c.validate_identity(resign(second))
    with pytest.raises(ValueError, match="Cross-stage"):
        c.assert_same_identity(first, second)


def test_resigned_epoch80_metadata_needs_independent_artifact_registration():
    identity = sample_identity()
    identity["purpose"] = "formal"
    identity["mapping"]["status"] = "FROZEN_CONCRETE_MAPPING"
    identity["checkpoint"].update(epoch=80, mode="formal", formal_eligible=True,
        evidence=dict(execution_manifest_sha256=c.EXECUTION_MANIFEST_SHA256,
                      training_result_sha256="f" * 64))
    identity["calibration"].update(number_of_calibration_images=64, source_verified=True)
    with pytest.raises(ValueError, match="Unregistered final EMA artifact"):
        c.validate_identity(resign(identity))


def test_real_archived_ema_is_diagnostic_not_final():
    # Real existing one-epoch training artifact, never promoted to a final model.
    path = c.FROZEN / "collected_terminal_20261013T142134Z_9cbac346/snapshot/remote/smoke/d9eeb9/seed_42/ema_final.pt"
    torch.set_num_threads(2)
    record, tensors = inspect_checkpoint(path, architecture="d9eeb9", seed=42)
    assert tensors and record["weights"] == "ema" and record["epoch"] == 1
    assert record["formal_eligible"] is False
    with pytest.raises(ValueError, match="epoch80"):
        inspect_checkpoint(path, architecture="d9eeb9", seed=42, purpose="formal")


def test_wrong_architecture_checkpoint_and_raw_payload_rejected(tmp_path):
    path = c.FROZEN / "collected_terminal_20261013T142134Z_9cbac346/snapshot/remote/smoke/d9eeb9/seed_42/ema_final.pt"
    with pytest.raises(ValueError, match="binding mismatch"):
        inspect_checkpoint(path, architecture="9ee366", seed=42)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["weights"] = "raw"
    bad = tmp_path / "raw.pt"
    torch.save(state, bad)
    with pytest.raises(ValueError, match="Explicit EMA"):
        inspect_checkpoint(bad, architecture="d9eeb9", seed=42)
