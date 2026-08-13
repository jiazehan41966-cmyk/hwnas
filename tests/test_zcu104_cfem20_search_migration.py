from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from hwnas_fpga.fourstage_operator import architecture_sha256, enumerate_cfem20
from hwnas_fpga.hardware.zcu104_operator_cost import (
    TEST_DATA_KIND,
    ZCU104CFEM20OperatorCostEstimator,
    canonical_sha256,
)
from hwnas_fpga.models import build_model
from hwnas_fpga.runtime import (
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.research_gates import zcu104_cfem20_search_gate_status
from hwnas_fpga.training import load_architecture_from_artifact
from hwnas_fpga.training.protocol_reporting import sha256_file
from hwnas_fpga.experiment import ExperimentTracker
from hwnas_fpga.interfaces import CandidateMetrics, SearchCandidate


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "artifacts/zcu104_full_campaign_v1/operator_cost_lut/fourstage_cfem20_v1/hardware_cost_lut.scaffold.json"
COVERAGE = ROOT / "artifacts/zcu104_full_campaign_v1/operator_cost_lut/fourstage_cfem20_v1/coverage_audit.json"


def _write_hashed(path: Path, payload: dict) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = canonical_sha256(unsigned)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _test_cost_files(tmp_path: Path) -> tuple[Path, Path]:
    lut = json.loads(SCAFFOLD.read_text(encoding="utf-8"))
    lut["data_kind"] = TEST_DATA_KIND
    lut["search_ready"] = True
    lut["status"] = "READY_TEST_ONLY_SYNTHETIC_9_OF_9"
    for index, row in enumerate(lut["rows"], start=1):
        row.update(
            {
                "source_commit": "1" * 40,
                "source_freeze_sha256": "2" * 64,
                "hls_config_sha256": "3" * 64,
                "bitstream_sha256": "4" * 64,
                "runtime_sha256": "5" * 64,
                "measurement_status": "ACCEPTED_FRESH_ZCU104_MEASUREMENT",
                "gate_status": "PASS",
            }
        )
        row["metrics"].update(
            {
                "latency_board_ms": index / 10,
                "lut": index * 100,
                "dsp": index * 2,
                "bram36": index,
                "p_dynamic_w": index / 20,
                "energy_dynamic_mj_per_op": index / 100,
                "wns_ns": 0.1,
                "tns_ns": 0.0,
            }
        )
    lut_path = tmp_path / "hardware_cost_lut.test_only_synthetic.json"
    _write_hashed(lut_path, lut)
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    coverage.update(
        {
            "search_ready": True,
            "status": "READY_TEST_ONLY_SYNTHETIC_20_OF_20",
            "operator_cost_payload_sha256": lut["payload_sha256"],
        }
    )
    coverage["measured_operator_coverage"] = {"covered": 9, "expected": 9, "status": "PASS"}
    coverage["measured_raw_candidate_coverage"] = {"covered": 20, "expected": 20, "status": "PASS"}
    for mapping in coverage["candidate_mappings"]:
        mapping["mapping_status"] = "PASS_MEASURED_COSTS_BOUND"
    coverage_path = tmp_path / "coverage_audit.test_only_synthetic.json"
    _write_hashed(coverage_path, coverage)
    return lut_path, coverage_path


def _search_space():
    config = {
        "search_space": {
            "stem_channels": 32,
            "stem_stride": 2,
            "post_stem_downsample_stride": 1,
            "stage_strides": [1, 2, 2, 1],
            "stage_channel_choices": [[16], [24], [32], [32]],
            "stage_depth_choices": [[1], [1], [1], [1]],
            "stage_block_choices": [
                [{"op": "conv", "kernel_size": 1, "expand_ratio": 1, "stride": 1}],
                [
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2},
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6, "stride": 2},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3, "stride": 2},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6, "stride": 2},
                ],
                [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2}],
                [
                    {"op": "skip", "kernel_size": 1, "expand_ratio": 1, "stride": 1},
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 1},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3, "stride": 1},
                    {"op": "dmdc_conv_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1},
                    {"op": "cfem_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1},
                ],
            ],
            "head_conv_channels": None,
            "head_channels": None,
        }
    }
    constraints = build_constraints(config)
    return config, constraints, build_search_space(
        config, image_size=224, input_channels=1, num_classes=8, constraints=constraints
    )


def _estimator(tmp_path: Path):
    lut_path, coverage_path = _test_cost_files(tmp_path)
    config, constraints, space = _search_space()
    estimator = ZCU104CFEM20OperatorCostEstimator(
        hardware_spec=build_hardware_spec({"hardware": {"board": "zcu104"}}),
        constraints=constraints,
        operator_cost_path=lut_path,
        coverage_path=coverage_path,
        allow_test_data=True,
    )
    return estimator, space, lut_path, coverage_path


def test_cfem20_all_20_architectures_map_to_two_of_nine_rows(tmp_path):
    estimator, space, _lut, _coverage = _estimator(tmp_path)
    observed = {}
    for candidate in enumerate_cfem20():
        estimate = estimator.estimate(candidate.architecture, space)
        mapping = estimator.mapping_by_sha[next(
            key for key, value in estimator.mapping_by_sha.items()
            if value["architecture_id"] == candidate.arch_id
        )]
        rows = [estimator.rows_by_key[mapping["stage2_cost_key"]], estimator.rows_by_key[mapping["stage4_cost_key"]]]
        assert estimate.latency_ms == pytest.approx(
            sum(float(row["metrics"]["latency_board_ms"]) for row in rows)
        )
        assert estimate.total_lut == sum(int(row["metrics"]["lut"]) for row in rows)
        observed[candidate.arch_id] = tuple(sorted((mapping["stage2_cost_key"], mapping["stage4_cost_key"])))
    assert len(observed) == 20
    assert len({key for keys in observed.values() for key in keys}) == 9
    assert estimator.get_lut_stats()["operator_cost_hits"] == 20


@pytest.mark.parametrize(
    "mutation",
    [
        "bad_hash",
        "missing_row",
        "negative_metric",
        "wrong_shape",
        "wrong_source",
        "wrong_mapping",
        "not_ready",
    ],
)
def test_operator_cost_input_fails_closed(tmp_path, mutation):
    lut_path, coverage_path = _test_cost_files(tmp_path)
    lut = json.loads(lut_path.read_text(encoding="utf-8"))
    if mutation == "bad_hash":
        lut["payload_sha256"] = "0" * 64
    elif mutation == "missing_row":
        lut["rows"].pop()
    elif mutation == "negative_metric":
        lut["rows"][0]["metrics"]["latency_board_ms"] = -1.0
    elif mutation == "wrong_shape":
        lut["rows"][0]["input_shape_chw"] = [16, 111, 112]
    elif mutation == "wrong_source":
        lut["rows"][0]["target"]["board"] = "AV7K325"
    elif mutation == "not_ready":
        lut["search_ready"] = False
    if mutation != "bad_hash":
        _write_hashed(lut_path, lut)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if mutation == "wrong_mapping":
        coverage["candidate_mappings"][0]["stage2_cost_key"] = coverage[
            "candidate_mappings"
        ][-1]["stage2_cost_key"]
    coverage["operator_cost_payload_sha256"] = lut["payload_sha256"]
    _write_hashed(coverage_path, coverage)
    config, constraints, _space = _search_space()
    with pytest.raises(ValueError):
        ZCU104CFEM20OperatorCostEstimator(
            hardware_spec=build_hardware_spec({"hardware": {"board": "zcu104"}}),
            constraints=constraints,
            operator_cost_path=lut_path,
            coverage_path=coverage_path,
            allow_test_data=True,
        )


def test_test_only_costs_require_explicit_opt_in(tmp_path):
    lut_path, coverage_path = _test_cost_files(tmp_path)
    config, constraints, _space = _search_space()
    with pytest.raises(ValueError, match="TEST_ONLY_SYNTHETIC"):
        ZCU104CFEM20OperatorCostEstimator(
            hardware_spec=build_hardware_spec({"hardware": {"board": "zcu104"}}),
            constraints=constraints,
            operator_cost_path=lut_path,
            coverage_path=coverage_path,
        )


def test_exact_shape_costs_reject_non_224_search_space(tmp_path):
    estimator, space, _lut, _coverage = _estimator(tmp_path)
    config, constraints, _ = _search_space()
    wrong_space = build_search_space(
        config, image_size=32, input_channels=1, num_classes=8, constraints=constraints
    )
    with pytest.raises(ValueError, match="image_size=224"):
        estimator.estimate(enumerate_cfem20()[0].architecture, wrong_space)


def test_search_artifact_rebuild_retrain_checkpoint_protocol_handoff(tmp_path):
    row = enumerate_cfem20()[0]
    candidate_path = tmp_path / "best_candidate.json"
    candidate = {
        "saved_at": "test-only",
        "candidate": {
            "arch_id": row.arch_id,
            "encoding": row.architecture.to_dict(),
            "architecture_sha256": architecture_sha256(row.architecture),
            "metrics": {"macro_f1": 0.0},
        },
        "test_only": True,
    }
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    architecture = load_architecture_from_artifact(candidate_path)
    model = build_model(architecture=architecture, num_classes=8)
    checkpoint_path = tmp_path / "final_best_model.test_only.pt"
    checkpoint = {
        "schema_version": 2,
        "source_candidate_file_sha256": sha256_file(candidate_path),
        "architecture": architecture.to_dict(),
        "architecture_sha256": candidate["candidate"]["architecture_sha256"],
        "training_config_sha256": "a" * 64,
        "data_split_sha256": "b" * 64,
        "seed": 42,
        "model_state_dict": model.state_dict(),
        "metrics": {"macro_f1": 0.0, "top1": 0.0},
        "test_only": True,
    }
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rebuilt = build_model(architecture=architecture, num_classes=8)
    rebuilt.load_state_dict(loaded["model_state_dict"])
    output = rebuilt(torch.zeros(1, 1, 224, 224))
    assert output.shape == (1, 8)
    assert len(sha256_file(candidate_path)) == 64
    assert len(sha256_file(checkpoint_path)) == 64


def test_search_tracker_persists_architecture_and_config_hashes(tmp_path):
    row = enumerate_cfem20()[0]
    tracker = ExperimentTracker(
        output_root=tmp_path,
        project_name="test-only",
        script_name="run_search",
        search_method="random",
        dataset_name="dummy",
        run_name="hash_handoff",
    )
    tracker.save_config({"project": {"seed": 42}}, {})
    tracker.save_best_candidate(
        SearchCandidate(
            arch_id=row.arch_id,
            encoding=row.architecture.to_dict(),
            metrics=CandidateMetrics(macro_f1=0.0),
        )
    )
    payload = json.loads(
        (tracker.checkpoints_dir / "best_candidate.json").read_text(encoding="utf-8")
    )
    assert payload["candidate"]["architecture_sha256"] == architecture_sha256(
        row.architecture
    )
    assert len(payload["search_config"]["sha256"]) == 64


def test_formal_search_config_waits_for_missing_formal_table():
    config = load_config(str(ROOT / "configs/search/zcu104_fourstage_cfem20_v1.yaml"))
    with pytest.raises(
        FileNotFoundError,
        match="WAITING_FOR_FORMAL_OPERATOR_COST_LUT.*hardware_cost_lut.formal.json",
    ):
        build_cost_estimator(
            config,
            hardware_spec=build_hardware_spec(config),
            constraints=build_constraints(config),
        )


def test_fresh_zcu104_lane_does_not_reuse_legacy_stage3_evidence():
    status = zcu104_cfem20_search_gate_status(
        {
            "cost_source": "zcu104_cfem20_exact_shape_operator_cost_v1",
            "data_kind": "FORMAL_ZCU104_MEASURED",
            "operator_rows": 9,
            "candidate_mappings": 20,
            "fallback_rate": 0.0,
            "operator_cost_file": "formal.json",
            "operator_cost_file_sha256": "1" * 64,
            "coverage_file": "coverage.json",
            "coverage_file_sha256": "2" * 64,
        }
    )
    assert status["status"] == "PASS_ZCU104_CFEM20_OPERATOR_COST"
    assert status["semantic_safe_search_space_cardinality"] == 20
    assert "AV7K325" in status["boundary"]


def test_test_only_cost_gate_is_smoke_only():
    status = zcu104_cfem20_search_gate_status(
        {
            "cost_source": "zcu104_cfem20_exact_shape_operator_cost_v1",
            "data_kind": "TEST_ONLY_SYNTHETIC",
            "operator_rows": 9,
            "candidate_mappings": 20,
            "fallback_rate": 0.0,
        }
    )
    assert status["status"] == "SMOKE_ONLY"
    assert status["may_start_new_claimable_search"] is False
