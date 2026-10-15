from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from hwnas_fpga.metrics_ledger import (
    METRIC_SCHEMA, append_metric_record, build_metric_record,
    read_metric_records, require_measured_metrics, validate_metric_record,
)


def measurement(value, **provenance):
    return {"value": value, "provenance": {"source": "valid_metrics.json", **provenance}}


def record(**overrides):
    fields = dict(task_type="detection", dataset="Cylinder2", network="D", seed=42,
                  stage="valid", identity={"run_id": "D_seed42_epoch80"},
                  measured={"map50_95": measurement(0.026), "recall": measurement(0.38)})
    fields.update(overrides)
    return build_metric_record(**fields)


def test_schema_completeness_and_no_fabricated_classification_or_hardware():
    event = record()
    assert len(event["metrics"]) == len(METRIC_SCHEMA)
    assert event["metrics"]["map50_95"]["status"] == "MEASURED"
    assert event["metrics"]["top1_accuracy"]["status"] == "NOT_APPLICABLE"
    assert event["metrics"]["macro_f1"]["status"] == "PENDING"
    assert event["metrics"]["board_latency_ms"]["value"] is None
    assert event["metrics"]["power_w"]["status"] == "PENDING"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), None, True, -0.1, 1.1])
def test_invalid_accuracy_is_rejected(value):
    with pytest.raises(ValueError):
        record(measured={"map50_95": measurement(value)})


def test_missing_provenance_unknown_metrics_and_wrong_units_rejected():
    with pytest.raises(ValueError, match="provenance.source"):
        record(measured={"map50_95": {"value": 0.01}})
    with pytest.raises(ValueError, match="bare scalar"):
        record(measured={"map50_95": 0.01})
    with pytest.raises(ValueError, match="Unknown metric"):
        record(measured={"magic_metric": measurement(1)})
    event = record()
    event["metrics"]["map50_95"]["unit"] = "%"
    with pytest.raises(ValueError, match="units"):
        validate_metric_record(event)
    with pytest.raises(ValueError, match="supplied unit"):
        record(measured={"map50_95": {**measurement(0.01), "unit": "%"}})


def test_pending_or_na_cannot_be_zero_and_keys_cannot_be_dropped():
    event = record()
    event["metrics"]["params"]["value"] = 0
    with pytest.raises(ValueError, match="never zero"):
        validate_metric_record(event)
    event = record()
    del event["metrics"]["params"]
    with pytest.raises(ValueError, match="Every schema metric"):
        validate_metric_record(event)


def test_detection_f1_requires_matching_not_ap_conversion():
    with pytest.raises(ValueError, match="matching_protocol"):
        record(measured={"macro_f1": measurement(0.2)})
    event = record(measured={"macro_f1": measurement(0.2, matching_protocol="IoU=.5,score=.25,class-aware")})
    assert event["metrics"]["macro_f1"]["value"] == 0.2
    with pytest.raises(ValueError, match="Top-1"):
        record(measured={"top1_accuracy": measurement(0.2)})


def test_classification_and_task_special_metrics():
    event = record(task_type="classification", dataset="NKSID", measured={
        "top1_accuracy": measurement(0.8),
        "per_class_f1": measurement({"mine": 0.6, "rock": 0.8}),
        "confusion_matrix": measurement([[3, 1], [1, 5]]),
        "tail_class_f1": measurement(0.6, tail_classes=["mine"], tail_definition="frozen-train-count")})
    assert event["metrics"]["map50_95"]["status"] == "NOT_APPLICABLE"
    with pytest.raises(ValueError, match="tail_classes"):
        record(task_type="classification", dataset="NKSID", measured={"tail_class_f1": measurement(0.5)})
    with pytest.raises(ValueError, match="size_definition"):
        record(measured={"small_target_recall": measurement(0.3)})


def test_image_quality_is_clean_reference_synthetic_only():
    with pytest.raises(ValueError, match="clean_reference"):
        record(measured={"psnr": measurement(25)})
    event = record(measured={"psnr": measurement(25, clean_reference=True,
        synthetic_noise=True, reference_manifest_sha256="reference_sha", noise_protocol="gaussian-v1", data_range=1)})
    assert event["metrics"]["psnr"]["value"] == 25
    event = record(identity={"run_id": "r", "clean_reference_available": False})
    assert event["metrics"]["psnr"]["status"] == "NOT_APPLICABLE"


def test_power_and_latency_cannot_come_from_estimates_or_macs():
    with pytest.raises(ValueError, match="board_measurement"):
        record(measured={"power_w": measurement(15, evidence_layer="vectorless_estimate")})
    with pytest.raises(ValueError, match="board_measurement"):
        record(measured={"board_latency_ms": measurement(20, evidence_layer="operator_sum")})
    event = record(measured={"power_w": measurement(4.2, evidence_layer="board_measurement",
        target_device="ZCU104", artifact_sha256="bit_sha", measurement_method="INA226",
        power_scope="rails-listed-in-source", idle_subtracted=False)})
    assert event["metrics"]["power_w"]["value"] == 4.2


def test_hv_requires_reference_and_timing_requires_route():
    with pytest.raises(ValueError, match="reference_point"):
        record(measured={"hypervolume": measurement(0.2, objectives=["map", "macs"],
            directions=["max", "min"], candidate_population="frozen72")})
    with pytest.raises(ValueError, match="place_and_route"):
        record(measured={"timing_met": measurement(True, evidence_layer="hls",
            target_device="ZCU104", artifact_sha256="a", tool_version="2024.1")})


def test_required_stage_metrics_cannot_remain_pending():
    require_measured_metrics(record(), ["map50_95", "recall"])
    with pytest.raises(ValueError, match="PENDING"):
        require_measured_metrics(record(), ["map50_95", "board_latency_ms"])
    with pytest.raises(ValueError, match="NOT_APPLICABLE"):
        require_measured_metrics(record(), ["top1_accuracy"])
    with pytest.raises(ValueError, match="Unknown required"):
        require_measured_metrics(record(), ["typo_metric"])


def test_feasible_rate_cannot_have_an_empty_denominator():
    with pytest.raises(ValueError, match="positive"):
        record(measured={"hardware_feasible_rate": measurement(0,
            denominator=0, feasibility_level="operator_sum", constraint_hash="h")})


def test_quantization_binds_comparisons_and_drop_sign():
    provenance = dict(checkpoint_sha256="weights", evaluation_protocol_hash="eval",
        split_manifest_sha256="split", numeric_contract="NUMERIC_CONTRACT_V1",
        calibration_manifest_sha256="calibration", definition="fp32_minus_int8")
    event = record(measured={"int8_map50_95_drop": measurement(-0.003, **provenance)})
    assert event["metrics"]["int8_map50_95_drop"]["value"] == -0.003
    with pytest.raises(ValueError, match="calibration_manifest"):
        record(measured={"int8_map50_95": measurement(0.02, checkpoint_sha256="weights",
            evaluation_protocol_hash="eval", split_manifest_sha256="split", numeric_contract="v1")})


def test_append_idempotence_conflict_and_resume(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = record()
    assert append_metric_record(path, first)
    assert not append_metric_record(path, record())
    conflicting = deepcopy(first)
    conflicting["metrics"]["map50_95"]["value"] = 0.5
    with pytest.raises(ValueError, match="Conflicting"):
        append_metric_record(path, conflicting)
    assert append_metric_record(path, record(identity={"run_id": "D_seed42_epoch81"}))
    assert len(read_metric_records(path)) == 2


def test_partial_crash_tail_is_preserved_then_recovered(tmp_path):
    path = tmp_path / "metrics.jsonl"
    assert append_metric_record(path, record())
    with path.open("ab") as handle:
        handle.write(b'{"schema_version": "sonar')
    with pytest.raises(ValueError, match="JSON"):
        read_metric_records(path)
    assert append_metric_record(path, record(identity={"run_id": "next_epoch"}))
    assert len(read_metric_records(path)) == 2
    tails = list(tmp_path.glob("*.recovery.*.bin"))
    assert len(tails) == 1
    assert tails[0].read_bytes() == b'{"schema_version": "sonar'


def test_complete_corrupt_line_fails_closed(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b"corrupted complete line\n")
    with pytest.raises(ValueError, match="Corrupt complete"):
        append_metric_record(path, record())
    assert path.read_bytes() == b"corrupted complete line\n"


def test_valid_final_line_without_newline_is_preserved(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(record()), encoding="utf-8")
    assert append_metric_record(path, record(identity={"run_id": "second"}))
    assert len(read_metric_records(path)) == 2


def test_concurrent_writers_do_not_lose_events(tmp_path):
    path = tmp_path / "metrics.jsonl"
    events = [record(identity={"run_id": f"run{i}"}) for i in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda event: append_metric_record(path, event), events))
    assert all(results)
    assert len(read_metric_records(path)) == 16


def test_separate_process_writers_do_not_lose_events(tmp_path):
    path = tmp_path / "multiprocess.jsonl"
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from hwnas_fpga.metrics_ledger import build_metric_record, append_metric_record
for index in range(4):
    event = build_metric_record(task_type='detection', dataset='Cylinder2', network='D',
        seed=42, stage='valid', identity={'run_id': sys.argv[3] + '_' + str(index)},
        measured={'recall': {'value': 0.5, 'provenance': {'source': 'valid.json'}}})
    append_metric_record(sys.argv[2], event)
"""
    source = str(Path(__file__).resolve().parents[1] / "src")
    processes = [subprocess.Popen([sys.executable, "-c", code, source, str(path), str(i)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for i in range(4)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, (stdout, stderr)
    assert len(read_metric_records(path)) == 16
