from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from architecture_transfer_common import validate_clean_manifest
from run_sctd_architecture_transfer import REGISTRATION as SCTD
from run_uatd_architecture_transfer import REGISTRATION as UATD


def test_uatd_identity_and_modality():
    manifest = validate_clean_manifest(UATD)["manifest"]
    assert UATD.dataset_id == "uatd_v3"
    assert UATD.modality == "MFLS"
    assert manifest["identity"]["dataset"] == "UATD v3"


def test_sctd_is_classic_v1_not_sctd3():
    card = (ROOT / "datasets/SCTD/metadata/dataset_card.md").read_text(encoding="utf-8")
    manifest = validate_clean_manifest(SCTD)["manifest"]
    assert SCTD.dataset_name == "classic SCTD / SCTD 1.0"
    assert SCTD.modality == "SSS"
    assert manifest["identity"]["is_sctd_3_0"] is False
    assert "This is not SCTD 3.0" in card

