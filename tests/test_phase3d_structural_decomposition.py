import inspect
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase3d_structural_common as phase3d
from hwnas_fpga.models.sonar_supernet import architecture_id
from run_sctd_architecture_transfer import REGISTRATION as SCTD
from run_uatd_architecture_transfer import REGISTRATION as UATD


def test_matched_architectures_are_frozen_and_differ_as_specified():
    mapping = phase3d.assert_matched_decomposition()
    assert mapping["D"]["architecture_id"] == "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3"
    assert mapping["B2"]["architecture_id"] == "P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
    assert mapping["B1"]["architecture_id"] == "P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
    for row in mapping.values():
        assert architecture_id(row["architecture"], spec=phase3d.common.V01_SPEC) == row["architecture_id"]
    assert mapping["B1"]["architecture"]["kernels"] == mapping["B2"]["architecture"]["kernels"]
    assert mapping["B1"]["architecture"]["output_combination"] == "P2_P3_P4_P5"
    assert mapping["B2"]["architecture"]["output_combination"] == mapping["D"]["architecture"]["output_combination"] == "P3_P4_P5"
    assert mapping["B2"]["architecture"]["kernels"] != mapping["D"]["architecture"]["kernels"]


def test_phase3d_does_not_retrain_d_or_b1():
    source = inspect.getsource(phase3d.train_b2) + inspect.getsource(phase3d.main)
    assert "train_b2" in source
    assert 'label == "B2"' in inspect.getsource(phase3d.smoke)
    assert "does not retrain D or B1" in inspect.getsource(phase3d.smoke)
    assert "resolve_phase3c_checkpoint" in inspect.getsource(phase3d.evaluate_validation)
    assert "phase3c_frozen" in inspect.getsource(phase3d.evaluate_validation)
    assert "phase3d_new" in inspect.getsource(phase3d.evaluate_validation)


def test_checkpoint_policy_is_epoch80_latest_not_best_valid():
    source = inspect.getsource(phase3d.train_b2)
    assert "range(start_epoch, 80)" in source
    assert 'latest = run_dir / "latest.pt"' in source
    assert "best.pt" not in source
    assert "best_valid" not in source
    assert '"checkpoint_policy": "epoch80/latest.pt"' in source


def test_phase3d_make_loader_rejects_test_before_dataloader():
    with pytest.raises(RuntimeError, match="PHASE3D_TEST_ACCESS_FORBIDDEN"):
        phase3d.phase3d_make_loader({}, "test", 0, 0, train=False)
    with pytest.raises(RuntimeError, match="PHASE3D_TEST_ACCESS_FORBIDDEN"):
        phase3d.guarded_create_yolo_detection_dataloader("unused", split="test")


def test_phase3d_sources_never_construct_test_loader():
    for name in ("train_b2", "evaluate_validation", "smoke", "phase3d_effective_config", "admit_phase3d_split"):
        source = inspect.getsource(getattr(phase3d, name))
        assert 'make_loader(config, "test"' not in source
        assert "split=\"test\"" not in source
        assert "split='test'" not in source
    config_source = inspect.getsource(phase3d.phase3d_effective_config)
    assert 'pop("test"' in config_source
    assert "PHASE3D_FORBIDDEN" in config_source


def test_admission_does_not_list_test_directory():
    source = inspect.getsource(phase3d.admit_phase3d_split)
    assert '("train", "valid")' in source
    assert '"test"' not in source.split("for split in", 1)[1].split(":", 1)[0]
    admission = phase3d.admit_phase3d_split(SCTD)
    assert admission["test_directory_listed"] is False
    assert admission["sha256"] == "7c20852975399d32fdb7d1d0fd36d986556a54826fb244585f3c9968d51667f9"
    uatd = phase3d.admit_phase3d_split(UATD)
    assert uatd["sha256"] == "937ac797c8a06d38d7248ebc8a7f987bd0acb88073302b86aad1d64806875141"
    config = phase3d.phase3d_effective_config(UATD)
    assert "test" not in config["dataset"]["split_paths"]
    assert config["evaluation_protocol"]["test_policy"] == "PHASE3D_FORBIDDEN"


def test_effect_additivity_and_decision_rules():
    effect = phase3d.summarize_effect((0.02, 0.01, 0.03))
    assert effect["mean"] == pytest.approx(0.02)
    assert effect["positive_seed_count"] == 3
    both = {
        "uatd_v3": {"mean": 0.02, "positive_seed_count": 3, "sample_std": 0.005},
        "classic_sctd_1_0": {"mean": 0.01, "positive_seed_count": 2, "sample_std": 0.004},
    }
    assert phase3d.classify_cross_dataset(both, supported="P2_CROSS_DATASET_SUPPORTED", domain="P2_DOMAIN_DEPENDENT", not_supported="P2_NOT_SUPPORTED_CROSS_DATASET") == "P2_CROSS_DATASET_SUPPORTED"
    mixed = {
        "uatd_v3": {"mean": 0.02, "positive_seed_count": 3, "sample_std": 0.001},
        "classic_sctd_1_0": {"mean": -0.01, "positive_seed_count": 0, "sample_std": 0.001},
    }
    assert phase3d.classify_cross_dataset(mixed, supported="P2_CROSS_DATASET_SUPPORTED", domain="P2_DOMAIN_DEPENDENT", not_supported="P2_NOT_SUPPORTED_CROSS_DATASET") == "P2_DOMAIN_DEPENDENT"
    none = {
        "uatd_v3": {"mean": -0.01, "positive_seed_count": 0, "sample_std": 0.001},
        "classic_sctd_1_0": {"mean": -0.02, "positive_seed_count": 1, "sample_std": 0.001},
    }
    assert phase3d.classify_cross_dataset(none, supported="P2_CROSS_DATASET_SUPPORTED", domain="P2_DOMAIN_DEPENDENT", not_supported="P2_NOT_SUPPORTED_CROSS_DATASET") == "P2_NOT_SUPPORTED_CROSS_DATASET"
    assert phase3d.interpret_structure("P2_CROSS_DATASET_SUPPORTED", "K5_BUNDLE_CROSS_DATASET_SUPPORTED", both) == "BOTH_SUPPORTED"
    assert phase3d.interpret_structure("P2_CROSS_DATASET_SUPPORTED", "K5_BUNDLE_NOT_SUPPORTED", both) == "P2_DOMINANT"
    assert phase3d.interpret_structure("P2_NOT_SUPPORTED_CROSS_DATASET", "K5_BUNDLE_CROSS_DATASET_SUPPORTED", both) == "K5_BUNDLE_DOMINANT"
    assert phase3d.interpret_structure("P2_DOMAIN_DEPENDENT", "K5_BUNDLE_DOMAIN_DEPENDENT", both) == "INTERACTION_OR_UNRESOLVED"
    assert phase3d.sonarop_p1_gate("K5_BUNDLE_NOT_SUPPORTED", none) == "NO"
    assert phase3d.sonarop_p1_gate("K5_BUNDLE_CROSS_DATASET_SUPPORTED", both) == "CONDITIONAL"
    strong = {
        "uatd_v3": {"mean": 0.03, "positive_seed_count": 3, "sample_std": 0.005},
        "classic_sctd_1_0": {"mean": 0.02, "positive_seed_count": 3, "sample_std": 0.004},
    }
    assert phase3d.sonarop_p1_gate("K5_BUNDLE_CROSS_DATASET_SUPPORTED", strong) == "YES"
    assert phase3d.stagewise_probe_needed("BOTH_SUPPORTED") == "NO"
    assert phase3d.stagewise_probe_needed("INTERACTION_OR_UNRESOLVED") == "NO"


def test_decompose_rejects_test_metrics_and_checks_additivity(tmp_path, monkeypatch):
    monkeypatch.setattr(phase3d, "output_dir_for", lambda registration: tmp_path / registration.dataset_id)

    def fake_row(label, value):
        return {
            "valid_metrics": {"Precision": 0.1, "Recall": 0.2, "mAP50": value + 0.1, "mAP50_95": value, "AP50": value + 0.1, "AP75": 0.0},
            "checkpoint_sha256": label,
            "checkpoint_source": "phase3c_frozen" if label != "B2" else "phase3d_new",
            "split_manifest_sha256": "split",
            "phase3d_test_accessed": False,
            "split": "valid",
        }

    matrix = {}
    for registration in (UATD, SCTD):
        for seed in (0, 1, 2):
            matrix[(registration.dataset_id, "D", seed)] = fake_row("D", 0.10 + 0.01 * seed)
            matrix[(registration.dataset_id, "B2", seed)] = fake_row("B2", 0.12 + 0.01 * seed)
            matrix[(registration.dataset_id, "B1", seed)] = fake_row("B1", 0.15 + 0.01 * seed)
    analysis = phase3d.decompose(matrix)
    assert analysis["phase3d_test_accessed"] is False
    assert analysis["p2_state"] == "P2_CROSS_DATASET_SUPPORTED"
    assert analysis["k5_state"] == "K5_BUNDLE_CROSS_DATASET_SUPPORTED"
    assert analysis["structural_interpretation"] == "BOTH_SUPPORTED"
    assert analysis["sonarop_p1_evidence_gate"] == "YES"
    uatd = analysis["datasets"]["uatd_v3"]["per_seed"][0]
    assert uatd["delta_P2"] + uatd["delta_K"] == pytest.approx(uatd["delta_TOTAL"])
    poisoned = dict(matrix[(UATD.dataset_id, "B1", 0)])
    poisoned["test_metrics"] = {"mAP50_95": 0.99}
    path = phase3d.validation_summary_path(UATD, "B1", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    phase3d.common.dump(path, poisoned)
    with pytest.raises(ValueError, match="test_metrics"):
        phase3d.load_validation_matrix()


def test_phase3c_valid_import_does_not_copy_test_metrics():
    admission = phase3d.admit_phase3d_split(UATD)
    config = phase3d.phase3d_effective_config(UATD)
    result = phase3d.import_phase3c_valid_epoch80(UATD, "D", 0, admission, config)
    assert result["split"] == "valid"
    assert result["phase3d_test_accessed"] is False
    assert "test_metrics" not in result
    assert result["checkpoint_source"] == "phase3c_frozen"
    assert result["evaluation_mode"] == "phase3c_epoch80_valid_cached"
    assert result["valid_metrics"]["mAP50_95"] == pytest.approx(0.3291985560836838)
    assert result["checkpoint_sha256"] == "285c5a54c8c468bd9c9f220ce845bfc971f078c9c484a93a18f844daaeaacf77"


def test_formal_requires_cuda(monkeypatch):
    monkeypatch.setattr(phase3d.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="READY_FOR_GPU_EXECUTION"):
        phase3d.require_cuda("cuda")
    with pytest.raises(RuntimeError, match="READY_FOR_GPU_EXECUTION"):
        phase3d.require_cuda("cpu")


def test_no_stagewise_or_sonarop_implementation():
    source = Path(phase3d.__file__).read_text(encoding="utf-8")
    assert "stage1 k3/k5" not in source
    assert "SonarOp P1" not in source or "does not implement SonarOp P1" in phase3d.protocol_text()
    assert "architecture re-search" not in inspect.getsource(phase3d.train_b2).lower()
    assert phase3d.B2_ID == "P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5"
    assert phase3d.FORMAL_SEEDS == (0, 1, 2)
