from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hwnas_fpga.fourstage_operator import (
    CFEM20_STAGE4_CHOICES,
    PENDING_STAGE4_OPERATORS,
    build_fourstage_architecture,
    enumerate_cfem20,
    validate_frozen_fourstage,
)
from hwnas_fpga.models import CFEMSonarBlock, build_model


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_ARCHITECTURE_IDS = [
    "fourstage_s2_k3_e3_s4_skip",
    "fourstage_s2_k3_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e3_s4_dmdc_conv_sonar",
    "fourstage_s2_k3_e3_s4_cfem_sonar",
    "fourstage_s2_k3_e6_s4_skip",
    "fourstage_s2_k3_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k3_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k3_e6_s4_dmdc_conv_sonar",
    "fourstage_s2_k3_e6_s4_cfem_sonar",
    "fourstage_s2_k5_e3_s4_skip",
    "fourstage_s2_k5_e3_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e3_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e3_s4_dmdc_conv_sonar",
    "fourstage_s2_k5_e3_s4_cfem_sonar",
    "fourstage_s2_k5_e6_s4_skip",
    "fourstage_s2_k5_e6_s4_mbconv_k3_e3",
    "fourstage_s2_k5_e6_s4_mbconv_k5_e3",
    "fourstage_s2_k5_e6_s4_dmdc_conv_sonar",
    "fourstage_s2_k5_e6_s4_cfem_sonar",
]


def test_cfem20_active_space_contains_cfem_and_keeps_msconv_pending():
    rows = enumerate_cfem20()
    assert CFEM20_STAGE4_CHOICES == (
        "skip",
        "mbconv_k3_e3",
        "mbconv_k5_e3",
        "dmdc_conv_sonar",
        "cfem_sonar",
    )
    assert len(rows) == 20
    assert [row.arch_id for row in rows] == EXPECTED_ARCHITECTURE_IDS
    assert len({row.arch_id for row in rows}) == 20
    assert all("fused_mbconv" not in row.arch_id for row in rows)
    assert all("ghost_bottleneck" not in row.arch_id for row in rows)
    assert all("msconv" not in row.arch_id for row in rows)
    assert PENDING_STAGE4_OPERATORS["msconv_sonar"]["implementation_status"] == (
        "pending_original_spec"
    )
    assert PENDING_STAGE4_OPERATORS["msconv_sonar"]["active_searchable"] is False
    for row in rows:
        validate_frozen_fourstage(row.architecture)


def test_cfem_stage4_block_preserves_fixed_shape_and_core_rates():
    block = CFEMSonarBlock(32, 32, stride=1).eval()
    assert block.dilation_rates == (2, 3, 5)
    with torch.no_grad():
        output = block(torch.zeros(2, 32, 28, 28))
    assert output.shape == (2, 32, 28, 28)


def test_cfem_rejects_non_stage4_adaptation():
    with pytest.raises(ValueError, match="32->32"):
        CFEMSonarBlock(24, 32, stride=2)


def test_cfem_candidate_builds_inside_full_fourstage_network():
    architecture = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="cfem_sonar",
    )
    model = build_model(architecture, num_classes=8).eval()
    assert isinstance(model.stages[3][0], CFEMSonarBlock)
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 224, 224))
    assert logits.shape == (1, 8)


def test_cfem20_freeze_script_writes_current_search_space(tmp_path):
    output_dir = tmp_path / "cfem20"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "freeze_fourstage_search_space_cfem20.py"),
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
        "candidate_manifest_20.json",
        "operator_definition.json",
        "search_space_audit.json",
    }
    assert {path.name for path in output_dir.glob("*.json")} == expected_files
    definition = json.loads(
        (output_dir / "search_space_definition.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "candidate_manifest_20.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (output_dir / "search_space_audit.json").read_text(encoding="utf-8")
    )
    assert definition["stage2_candidate_count"] == 4
    assert definition["stage4_candidate_count"] == 5
    assert definition["raw_network_count"] == 20
    assert definition["pending_stage4_operators"][0]["operator_name"] == "msconv_sonar"
    assert definition["pending_stage4_operators"][0]["active_space_status"] == (
        "pending_original_spec"
    )
    assert definition["pending_stage4_operators"][0]["active_searchable"] is False
    cfem = {
        row["operator_name"]: row
        for row in definition["stage4_candidates"]
        if row.get("operator_name") == "cfem_sonar"
    }["cfem_sonar"]
    assert cfem["source_type"] == "literature_operator"
    assert cfem["source_task"] == "forward_looking_sonar_object_detection"
    assert cfem["sonar_specific"] is True
    assert cfem["candidate_role"] == "sonar_context_multiscale_candidate"
    assert cfem["dilation_rates"] == [2, 3, 5]
    assert cfem["implementation_status"] == "implemented"
    assert manifest["raw_network_count"] == 20
    assert [row["architecture_id"] for row in manifest["rows"]] == (
        EXPECTED_ARCHITECTURE_IDS
    )
    assert audit["status"] == "PASS"
    assert all(audit["checks"].values())
    assert audit["claim_boundary"]["training"] == "NOT_RUN"
    assert audit["claim_boundary"]["nas_search"] == "NOT_RUN"
    assert audit["claim_boundary"]["hls"] == "NOT_RUN"
    assert audit["claim_boundary"]["route"] == "NOT_RUN"
    assert audit["claim_boundary"]["power"] == "NOT_MEASURED"
