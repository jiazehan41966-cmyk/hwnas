from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hwnas_fpga.fourstage_operator import (
    DMDC16_STAGE4_CHOICES,
    EXCLUDED_FROM_CURRENT_STAGE4_SPACE,
    PENDING_STAGE4_OPERATORS,
    build_fourstage_architecture,
    enumerate_dmdc16,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import DMDCConvSonarBlock, build_model


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ARCHITECTURE_IDS = [
    "fourstage_s2_k3_e3_s4_skip",
    "fourstage_s2_k3_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e3_s4_dmdc_conv_sonar",
    "fourstage_s2_k3_e6_s4_skip",
    "fourstage_s2_k3_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e6_s4_dmdc_conv_sonar",
    "fourstage_s2_k5_e3_s4_skip",
    "fourstage_s2_k5_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e3_s4_dmdc_conv_sonar",
    "fourstage_s2_k5_e6_s4_skip",
    "fourstage_s2_k5_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e6_s4_dmdc_conv_sonar",
]


def test_dmdc16_active_space_has_only_current_stage4_candidates():
    rows = enumerate_dmdc16()
    assert len(rows) == 16
    assert DMDC16_STAGE4_CHOICES == (
        "skip",
        "mbconv_k3_e3",
        "mbconv_k5_e3",
        "dmdc_conv_sonar",
    )
    assert [row.arch_id for row in rows] == EXPECTED_ARCHITECTURE_IDS
    assert len({row.arch_id for row in rows}) == 16
    assert "msconv_sonar" in PENDING_STAGE4_OPERATORS
    assert all("msconv" not in row.arch_id for row in rows)
    assert all("fused_mbconv" not in row.arch_id for row in rows)
    assert all("ghost_bottleneck" not in row.arch_id for row in rows)
    assert EXCLUDED_FROM_CURRENT_STAGE4_SPACE == {
        "fused_mbconv_e3": "excluded_from_current_space",
        "ghost_bottleneck": "excluded_from_current_space",
    }
    for row in rows:
        validate_frozen_fourstage(row.architecture)


def test_dmdc_stage4_block_preserves_fixed_stage4_shape():
    block = DMDCConvSonarBlock(32, 32, stride=1).eval()
    assert block.dilation_rates == (1, 3, 5)
    assert [branch.dilation_rate for branch in block.scale_branches] == [1, 3, 5]
    with torch.no_grad():
        output = block(torch.zeros(2, 32, 28, 28))
    assert output.shape == (2, 32, 28, 28)


def test_dmdc_rejects_non_stage4_adaptation():
    with pytest.raises(ValueError, match="32->32"):
        DMDCConvSonarBlock(24, 32, stride=2)


def test_dmdc_candidate_builds_inside_full_fourstage_network():
    architecture = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="dmdc_conv_sonar",
    )
    model = build_model(architecture, num_classes=8).eval()
    assert isinstance(model.stages[3][0], DMDCConvSonarBlock)
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 224, 224))
    assert logits.shape == (1, 8)


def test_dmdc16_freeze_script_writes_current_search_space(tmp_path):
    output_dir = tmp_path / "dmdc16"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_fourstage_search_space_dmdc16.py"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    expected_files = {
        "search_space_definition.json",
        "candidate_manifest_16.json",
        "operator_definition.json",
        "search_space_audit.json",
    }
    assert {path.name for path in output_dir.glob("*.json")} == expected_files
    definition = json.loads(
        (output_dir / "search_space_definition.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "candidate_manifest_16.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (output_dir / "search_space_audit.json").read_text(encoding="utf-8")
    )
    assert definition["stage2_candidate_count"] == 4
    assert definition["stage4_candidate_count"] == 4
    assert definition["candidate_count"] == 16
    assert definition["pending_stage4_operators"][0]["operator_name"] == "msconv_sonar"
    assert definition["pending_stage4_operators"][0]["active_space_status"] == (
        "pending_adaptation_review"
    )
    excluded = {
        row["operator_name"]: row["active_space_status"]
        for row in definition["excluded_from_current_space"]
    }
    assert excluded == {
        "fused_mbconv_e3": "excluded_from_current_space",
        "ghost_bottleneck": "excluded_from_current_space",
    }
    assert manifest["candidate_count"] == 16
    assert [row["architecture_id"] for row in manifest["rows"]] == (
        EXPECTED_ARCHITECTURE_IDS
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["claim_boundary"]["training"] == "NOT_RUN"
    assert audit["claim_boundary"]["hls"] == "NOT_RUN"
    assert audit["claim_boundary"]["route"] == "NOT_RUN"
    assert audit["claim_boundary"]["power"] == "NOT_MEASURED"
