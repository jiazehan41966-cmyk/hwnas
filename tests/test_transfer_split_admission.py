from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from architecture_transfer_common import validate_clean_manifest
from run_sctd_architecture_transfer import REGISTRATION as SCTD
from run_uatd_architecture_transfer import REGISTRATION as UATD


def test_clean_manifests_are_admitted_and_duplicate_safe():
    for registration in (UATD, SCTD):
        admission = validate_clean_manifest(registration)
        manifest = admission["manifest"]
        assert manifest["duplicate_groups_cross_derived_split"] == 0
        assert manifest["test_membership_frozen"] is True
        assert manifest["model_metrics_used"] is False
        assert manifest["raw_data_modified"] is False


def test_uatd_group_structure_unknown_and_official_test_preserved():
    manifest = validate_clean_manifest(UATD)["manifest"]
    assert manifest["policy"]["acquisition_group_status"] == "GROUP_STRUCTURE_UNKNOWN"
    assert manifest["policy"]["official_test_preserved"] is True
    assert len(manifest["excluded_source_items"]) == 2


def test_sctd_v2_has_all_classes_in_each_split():
    manifest = validate_clean_manifest(SCTD)["manifest"]
    for rows in manifest["members"].values():
        assert {int(cls) for row in rows for cls in row["classes"]} == {0, 1, 2}

