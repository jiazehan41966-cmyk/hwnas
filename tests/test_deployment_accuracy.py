import json

from hwnas_fpga.hardware.deployment_accuracy import audit_accuracy_retention


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_accuracy_retention_passes_with_complete_board_predictions(tmp_path):
    fp32 = tmp_path / "fp32.json"
    int8 = tmp_path / "int8.json"
    board = tmp_path / "board.json"
    predictions = tmp_path / "board_predictions.jsonl"
    _write_json(fp32, {"metrics": {"macro_f1": 0.91, "top1": 0.93}})
    _write_json(int8, {"metrics": {"macro_f1": 0.90, "top1": 0.925}})
    _write_json(
        board,
        {
            "classification": {"macro_f1": 0.895, "top1": 0.915},
            "board": {
                "expected_samples": 2,
                "observed_samples": 2,
                "crc_error_count": 0,
                "status_error_count": 0,
                "argmax_mismatch_count": 0,
            },
        },
    )
    predictions.write_text(
        '{"sample_id":"a","argmax":1}\n{"sample_id":"b","argmax":0}\n',
        encoding="utf-8",
    )

    result = audit_accuracy_retention(
        fp32_summary=fp32,
        int8_summary=int8,
        board_summary=board,
        board_predictions_jsonl=predictions,
        macro_f1_drop_threshold=0.02,
        top1_drop_threshold=0.02,
    )

    assert result["status"] == "PASS"
    assert result["deltas"]["delta_board_macro_f1"] == -0.015000000000000013
    assert result["board_quality"]["sample_coverage"] == 1.0
    assert result["board_predictions_jsonl"]["line_count"] == 2


def test_accuracy_retention_fails_on_board_mismatch_and_short_coverage(tmp_path):
    fp32 = tmp_path / "fp32.json"
    int8 = tmp_path / "int8.json"
    board = tmp_path / "board.json"
    predictions = tmp_path / "board_predictions.jsonl"
    _write_json(fp32, {"macro_f1": 0.91, "top1": 0.93})
    _write_json(int8, {"macro_f1": 0.90, "top1": 0.925})
    _write_json(
        board,
        {
            "macro_f1": 0.86,
            "top1": 0.90,
            "expected_samples": 4,
            "observed_samples": 3,
            "crc_error_count": 0,
            "status_error_count": 1,
            "argmax_mismatch_count": 1,
        },
    )
    predictions.write_text(
        '{"sample_id":"a","argmax":1}\n{"sample_id":"b","argmax":0}\n',
        encoding="utf-8",
    )

    result = audit_accuracy_retention(
        fp32_summary=fp32,
        int8_summary=int8,
        board_summary=board,
        board_predictions_jsonl=predictions,
    )

    assert result["status"] == "FAIL"
    assert result["gates"]["board_macro_f1_retained"] is False
    assert result["gates"]["board_sample_coverage_100pct"] is False
    assert result["gates"]["board_status_error_count_zero"] is False
    assert result["gates"]["board_argmax_mismatch_count_zero"] is False
    assert result["gates"]["board_per_sample_predictions_saved"] is False
