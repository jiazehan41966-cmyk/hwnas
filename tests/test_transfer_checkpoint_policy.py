import inspect
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import architecture_transfer_common as transfer


def test_formal_checkpoint_is_latest_epoch80_not_best_valid():
    source = inspect.getsource(transfer.formal_run)
    assert 'latest = run_dir / "latest.pt"' in source
    assert '"checkpoint_policy": "epoch80/latest.pt"' in source
    assert 'best.pt' not in source
    assert 'best_valid' not in source


def test_formal_seeds_are_frozen():
    assert transfer.FORMAL_SEEDS == (0, 1, 2)


def test_legacy_or_foreign_formal_result_is_rejected():
    binding = {
        "dataset_id": "classic_sctd_1_0",
        "model": "baseline",
        "seed": 0,
        "architecture": {"architecture_id": "FrozenBaseline_P3_P4_P5_all_k3"},
        "split_manifest_sha256": "clean-split",
    }
    legacy = {
        "artifact_type": "sctd_architecture_transfer_summary",
        "model": "baseline",
        "seed": 0,
        "architecture_id": "FrozenBaseline_P3_P4_P5_all_k3",
        "checkpoint_policy": "best-valid",
        "split_manifest_sha256": None,
        "test_evaluated": True,
        "test_used_for_selection": False,
    }
    try:
        transfer.validate_formal_result(legacy, binding)
    except ValueError as error:
        assert "foreign or legacy formal result rejected" in str(error)
    else:
        raise AssertionError("legacy result unexpectedly admitted")


def test_phase3c_formal_result_requires_exact_binding():
    binding = {
        "dataset_id": "uatd_v3",
        "model": "nas",
        "seed": 2,
        "architecture": {"architecture_id": transfer.B1_ID},
        "split_manifest_sha256": "clean-split",
    }
    result = {
        "artifact_type": "phase3c_architecture_transfer_result",
        "dataset_id": "uatd_v3",
        "model": "nas",
        "seed": 2,
        "architecture_id": transfer.B1_ID,
        "checkpoint_policy": "epoch80/latest.pt",
        "split_manifest_sha256": "clean-split",
        "test_evaluated": True,
        "test_used_for_selection": False,
        "binding": dict(binding),
    }
    transfer.validate_formal_result(result, binding)
