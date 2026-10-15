"""Adversarial transport/confirmation checks; every generated record is synthetic.

The compact graph only exercises byte transport. It is not the real detector,
and no fixture may become a formal checkpoint, quantization or board result.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.sonar_192_chain.contract import digest, make_identity, model_contract, sha256_file
from tools.sonar_192_chain.model import inspect_checkpoint
from tools.sonar_192_chain.quantization import NUMERIC_CONTRACT, NUMERIC_CONTRACT_HASH
from tools.sonar_192_chain.confirmation import (
    METRIC, SCHEMA, confirm_candidates, reference, replay_confirmation, validate_validation_contract, require_registered_evaluation,
)
from tools.sonar_192_chain.handoff import (
    ABI, BOARD_SCHEMA, evaluate_measurements, inspect_export, inspect_parameter_package,
    prepare_parameter_package, validate_board_return,
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    return reference(path)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def identity_fixture(folder, seed=42, epoch=80):
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "ema_final.pt"
    torch.save({"epoch": epoch, "weights": "ema", "inference_only": True,
                "model_state_dict": {"synthetic_test_tensor": torch.tensor([seed * .001])},
                "ema_updates": epoch * 83, "test_evaluated": False,
                "binding": {"architecture_id": "9ee366", "seed": seed, "mode": "formal",
                            "head": "quality_direct6_CIoU", "recipe": "C01_plus_EMA",
                            "validation_weights": "ema", "test_evaluated": False}}, checkpoint)
    binding, _ = inspect_checkpoint(checkpoint, architecture="9ee366", seed=seed, purpose="diagnostic")
    identity = make_identity("9ee366", checkpoint=binding, profile="W8A8",
        mapping={"implementation_id": "synthetic_transport_fixture", "PE": 1, "SIMD": 1,
                 "buffer_profile": "synthetic", "numeric_contract_sha256": NUMERIC_CONTRACT_HASH,
                 "interface_contract_sha256": digest({"interface": "synthetic_transport"}), "status": "PROPOSED"},
        calibration={"sha256": digest({"calibration": "synthetic"}), "split": "train", "test_accessed": False,
                     "number_of_calibration_images": 1, "source_verified": False}, purpose="diagnostic")
    return identity, checkpoint


def export_fixture(folder, identity):
    folder.mkdir(parents=True, exist_ok=True)
    heads = [f"head{i}" for i in range(4)]
    nodes = [{"id": name, "op": "conv", "inputs": ["image"], "input_shapes": [[1, 2, 2]],
              "output_shape": [6, 1, 1], "Cin": 1, "Cout": 6, "kernel": 2, "stride": 1,
              "padding": 0, "groups": 1} for name in heads]
    graph = {key: identity[key] for key in ("space_version", "architecture_sha256", "full_contract_sha256")}
    graph.update(nodes=nodes, input_shape=[1, 2, 2], outputs=dict(zip(("P2", "P3", "P4", "P5"), heads)))
    graph["numeric_graph_sha256"] = digest(graph)
    arrays, entries, parameter_hashes = {}, [], {}
    for name in heads:
        parameter_hashes[name] = {}
        for kind, value in (("weight", np.ones((6, 1, 2, 2), dtype=np.int8)),
                            ("bias", np.zeros(6, dtype=np.int32))):
            key = f"parameter_{len(arrays)}"
            arrays[key] = value
            entries.append({"node": name, "kind": kind, "key": key, "dtype": value.dtype.str,
                            "shape": list(value.shape), "sha256": array_sha(value)})
            parameter_hashes[name][kind + "_sha256"] = array_sha(value)
    parameter_hash = digest({"parameters": parameter_hashes, "luts": {}})
    np.savez(folder / "integer_parameters.npz", **arrays)
    save(folder / "parameter_index.json", {"parameters": entries})
    profile = {key: identity[key] for key in ("space_version", "architecture_sha256", "full_contract_sha256")}
    profile.update(precision_profile="W8A8", checkpoint_binding=identity["checkpoint"], identity=identity,
                   identity_sha256=identity["identity_sha256"], calibration=identity["calibration"],
                   original_model_state_sha256=identity["checkpoint"]["model_state_sha256"],
                   numeric_graph_sha256=graph["numeric_graph_sha256"], numeric_contract=NUMERIC_CONTRACT,
                   numeric_contract_hash=NUMERIC_CONTRACT_HASH, integer_parameters_sha256=parameter_hash,
                   source_sha256={str(Path(__file__).resolve()): sha256_file(__file__)}, evidence_kind="synthetic_diagnostic")
    values = {"image": np.ones((1, 1, 2, 2), dtype=np.int8)}
    values.update({name: np.full((1, 6, 1, 1), 4, dtype=np.int8) for name in heads})
    raw, ref_rows, witnesses = {}, [], []
    for name, value in values.items():
        key = f"node_{len(raw)}"
        raw[key] = value
        ref_rows.append({"node": name, "key": key, "dtype": value.dtype.str, "shape": list(value.shape),
                         "sha256": array_sha(value)})
        witnesses.append({"node": name, "shape": list(value.shape), "element_count": value.size,
                          "integer_sha256": array_sha(value), "fake_quantized_sha256": array_sha(value), "mismatch_count": 0})
    np.savez(folder / "reference_outputs.npz", **raw)
    save(folder / "reference_index.json", {"nodes": ref_rows, "input_images_sha256": digest({"synthetic_input": 1}),
         "numeric_graph_sha256": graph["numeric_graph_sha256"], "integer_parameters_sha256": parameter_hash})
    save(folder / "software_parity_witness.json", {"mismatches": 0, "rows": witnesses, "input_images": 1,
         "numeric_graph_sha256": graph["numeric_graph_sha256"], "integer_parameters_sha256": parameter_hash,
         "identity_sha256": identity["identity_sha256"],
         "original_model_state_sha256": identity["checkpoint"]["model_state_sha256"]})
    save(folder / "numeric_graph.json", graph)
    save(folder / "profile.json", profile)
    rewrite_export_manifest(folder)
    return folder / "artifact_manifest.json"


def rewrite_export_manifest(folder):
    save(folder / "artifact_manifest.json", {"files": {
        path.name: sha256_file(path) for path in folder.iterdir() if path.name != "artifact_manifest.json"}})


def record_fixture():
    return [{"image": f"valid/images/synthetic_{label}.png",
             "prediction": {"boxes": [[0., 0., 10., 10.]], "labels": [label], "scores": [.9]},
             "target": {"boxes": [[0., 0., 10., 10.]], "labels": [label]}} for label in (0, 1)]


def confirmation_fixture(folder):
    records = record_fixture()
    protocol = {"schema_version": SCHEMA, "seeds": [42, 43, 44], "profiles": ["W8A8"],
                "metric": METRIC, "selection_policy": "epoch80_ema", "num_classes": 2,
                "validation_count": 2, "validation_images_sha256": digest(sorted(row["image"] for row in records)),
                "validation_targets_sha256": digest([{"image": row["image"], "target": row["target"]} for row in records]),
                "absolute_drop": .003, "relative_drop": .20, "improved_seeds": 2, "evidence_kind": "synthetic_diagnostic"}
    receipts, identities = [], []
    for seed in (42, 43, 44):
        sub = folder / str(seed)
        identity, checkpoint = identity_fixture(sub, seed)
        export = export_fixture(sub / "ptq", identity)
        fp32 = save(sub / "fp32.json", records)
        quant = save(sub / "quantized.json", records)
        common = {"identity_sha256": identity["identity_sha256"], "checkpoint_sha256": identity["checkpoint"]["sha256"],
                  "weights": "ema", "epoch": 80, "split": "valid"}
        receipt = {"schema_version": SCHEMA, "evidence_kind": "synthetic_diagnostic", "identity": identity,
                   "checkpoint": reference(checkpoint), "metric": METRIC, "selection_policy": "epoch80_ema",
                   "test_evaluated": False, "fp32_records": fp32,
                   "fp32_evaluation_binding": {**common, "records_sha256": fp32["sha256"]},
                   "stable_epoch_records": {str(epoch): fp32 for epoch in range(76, 81)},
                   "profiles": {"W8A8": {"records": quant, "export_manifest": reference(export),
                       "evaluation_binding": {**common, "records_sha256": quant["sha256"],
                                              "export_manifest_sha256": sha256_file(export)}}}}
        receipts.append(save(sub / "confirmation_receipt.json", receipt))
        identities.append(identity)
    return identities[0], receipts, protocol


def board_fixture(folder, candidate, loader_path):
    folder.mkdir(parents=True, exist_ok=True)
    loader = json.loads(loader_path.read_text())
    bit = folder / "synthetic_not_programmable.bit"
    bit.write_bytes(b"SYNTHETIC_DIAGNOSTIC_NOT_A_BITSTREAM")
    bindings = {**loader["bindings"], "loader_manifest_sha256": sha256_file(loader_path), "bitstream_sha256": sha256_file(bit)}
    index = json.loads(Path(loader["reference_index"]["path"]).read_text())
    index["bindings"] = bindings
    raw = folder / "raw.npz"
    raw.write_bytes(Path(loader["reference_outputs"]["path"]).read_bytes())
    impl = save(folder / "implementation.json", {"abi": ABI, "bindings": bindings, "evidence_kind": "synthetic_diagnostic"})
    capture = save(folder / "capture.json", {"bindings": bindings, "abi_observed": ABI, "device_id": "synthetic",
                   "loaded_parameter_binary_sha256": loader["binary"]["sha256"], "evidence_kind": "synthetic_diagnostic"})
    receipt = {"schema_version": BOARD_SCHEMA, "bindings": bindings, "device_id": "synthetic", "test_evaluated": False,
               "bitstream": reference(bit), "implementation_manifest": impl, "loader_capture": capture,
               "raw_outputs": reference(raw), "output_index": save(folder / "index.json", index),
               "evidence_kind": "synthetic_diagnostic"}
    save(folder / "board_receipt.json", receipt)
    return folder / "board_receipt.json"


def test_binary_roundtrip_and_diagnostic_board_is_never_formal(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    report = prepare_parameter_package(identity, export, output=tmp_path / "package")
    assert report["formal_eligible"] is False and report["hardware_compatible"] is False
    loader_path = Path(report["loader_manifest"]["path"])
    loader, _ = inspect_parameter_package(identity, loader_path)
    assert loader["abi"] == ABI and loader["transport_only"]
    board = board_fixture(tmp_path / "board", identity, loader_path)
    result = validate_board_return(identity, loader_path, board)
    assert result["status"] == "FULL_GRAPH_INTEGER_PARITY_PASS" and result["mismatches"] == 0
    assert not result["formal_eligible"] and not result["hardware_verified"]
    assert "FROZEN_CLOCK_DEVICE_PROTOCOL_MISSING" in result["formal_blockers"]
    assert not any(result[key] for key in ("valid_accuracy_verified", "latency_verified", "power_verified"))


def test_rehashing_modified_parameters_cannot_bypass_export_binding(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    file = export.parent / "integer_parameters.npz"
    with np.load(file) as data:
        arrays = {key: data[key].copy() for key in data.files}
    arrays["parameter_0"].flat[0] = 99
    np.savez(file, **arrays)
    rewrite_export_manifest(export.parent)
    with pytest.raises(ValueError, match="parameter byte hash"):
        inspect_export(identity, export)


def test_ddr_overlap_rejected_after_layout_and_loader_rehash(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    report = prepare_parameter_package(identity, export, output=tmp_path / "package")
    loader_path = Path(report["loader_manifest"]["path"])
    loader = json.loads(loader_path.read_text())
    ddr_path = Path(loader["ddr_layout"]["path"])
    ddr = json.loads(ddr_path.read_text())
    ddr["parameters"][1]["offset"] = 0
    loader["ddr_layout"] = save(ddr_path, ddr)
    save(loader_path, loader)
    with pytest.raises(ValueError, match="overlap"):
        inspect_parameter_package(identity, loader_path)


def test_loader_cannot_replace_the_quantization_raw_reference(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    package = prepare_parameter_package(identity, export, output=tmp_path / "package")
    loader_path = Path(package["loader_manifest"]["path"])
    loader = json.loads(loader_path.read_text())
    changed = tmp_path / "unrelated_raw_outputs.npz"
    with np.load(loader["reference_outputs"]["path"]) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["node_1"].flat[0] += 1
    np.savez(changed, **arrays)
    loader["reference_outputs"] = reference(changed)
    save(loader_path, loader)
    with pytest.raises(ValueError, match="raw reference source differs"):
        inspect_parameter_package(identity, loader_path)


@pytest.mark.parametrize("fault", ["one_integer_changed", "missing_node", "wrong_weight_binding", "fake_PASS_only"])
def test_board_returns_rechecked_from_raw_bytes(tmp_path, fault):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    package = prepare_parameter_package(identity, export, output=tmp_path / "package")
    loader = Path(package["loader_manifest"]["path"])
    board = board_fixture(tmp_path / "board", identity, loader)
    receipt = json.loads(board.read_text())
    if fault == "fake_PASS_only":
        save(board, {"schema_version": BOARD_SCHEMA, "status": "PASS", "mismatches": 0, "test_evaluated": False})
        with pytest.raises((ValueError, KeyError)):
            validate_board_return(identity, loader, board)
        return
    if fault == "wrong_weight_binding":
        receipt["bindings"]["weight_sha256"] = "a" * 64
        save(board, receipt)
        with pytest.raises(ValueError, match="binding differs"):
            validate_board_return(identity, loader, board)
        return
    index_path = Path(receipt["output_index"]["path"])
    index = json.loads(index_path.read_text())
    if fault == "missing_node":
        index["nodes"].pop()
        receipt["output_index"] = save(index_path, index)
        save(board, receipt)
        with pytest.raises(ValueError, match="every graph node"):
            validate_board_return(identity, loader, board)
        return
    raw_path = Path(receipt["raw_outputs"]["path"])
    with np.load(raw_path) as payload:
        arrays = {key: payload[key].copy() for key in payload.files}
    arrays["node_1"].flat[0] += 1
    index["nodes"][1]["sha256"] = array_sha(arrays["node_1"])
    np.savez(raw_path, **arrays)
    receipt["raw_outputs"] = reference(raw_path)
    receipt["output_index"] = save(index_path, index)
    save(board, receipt)
    result = validate_board_return(identity, loader, board)
    assert result["status"] == "INTEGER_PARITY_FAILED" and result["mismatches"] == 1
    assert not result["hardware_verified"]


def test_three_seed_confirmation_recomputes_and_separates_checkpoint_metric(tmp_path):
    identity, receipts, protocol = confirmation_fixture(tmp_path)
    result = confirm_candidates(identity, receipts, protocol)
    assert result["status"] == "PAIRED_ADMISSION_PASS" and not result["formal_eligible"]
    assert result["points"][0]["mean_mAP50_95"] == pytest.approx(1.)
    assert all(row["fp32_epoch80_ema_mAP50_95"] == pytest.approx(1.)
               and row["fp32_epoch76_80_ema_mean_mAP50_95"] == pytest.approx(1.) for row in result["rows"])
    assert replay_confirmation(result) == result
    with pytest.raises(ValueError, match="three seed"):
        confirm_candidates(identity, receipts[:2], protocol)


def test_claimed_accuracy_cannot_replace_raw_quantized_records(tmp_path):
    identity, receipts, protocol = confirmation_fixture(tmp_path)
    for ref in receipts:
        path = Path(ref["path"])
        receipt = json.loads(path.read_text())
        item = receipt["profiles"]["W8A8"]
        bad = record_fixture()
        for row in bad:
            row["prediction"] = {"boxes": [], "scores": [], "labels": []}
        item["records"] = save(Path(item["records"]["path"]), bad)
        item["evaluation_binding"]["records_sha256"] = item["records"]["sha256"]
        item["claimed_mAP50_95"] = 1.
        ref.update(save(path, receipt))
    result = confirm_candidates(identity, receipts, protocol)
    assert result["status"] == "PAIRED_ADMISSION_REJECTED"
    assert result["points"][0]["mean_mAP50_95"] == 0.


def test_duplicate_seed_and_epoch_substitution_rejected(tmp_path):
    identity, receipts, protocol = confirmation_fixture(tmp_path)
    with pytest.raises(ValueError, match="Duplicate or missing"):
        confirm_candidates(identity, [receipts[0], receipts[0], receipts[2]], protocol)
    path = Path(receipts[0]["path"])
    receipt = json.loads(path.read_text())
    receipt["fp32_evaluation_binding"]["epoch"] = "mean76_80"
    receipts[0] = save(path, receipt)
    with pytest.raises(ValueError, match="FP32 evaluation binding"):
        confirm_candidates(identity, receipts, protocol)


def test_selected_identity_must_belong_to_confirmed_seed_receipts(tmp_path):
    identity, receipts, protocol = confirmation_fixture(tmp_path / "paired")
    unconfirmed, _ = identity_fixture(tmp_path / "unconfirmed")
    assert unconfirmed["identity_sha256"] != identity["identity_sha256"]
    with pytest.raises(ValueError, match="not one of the confirmed"):
        confirm_candidates(unconfirmed, receipts, protocol)


def test_synthetic_candidate_cannot_be_promoted_to_formal_by_flag(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    changed = deepcopy(identity)
    changed["purpose"] = "formal"
    changed["mapping"]["status"] = "FROZEN_CONCRETE_MAPPING"
    changed["identity_sha256"] = digest({key: value for key, value in changed.items() if key != "identity_sha256"})
    with pytest.raises(ValueError, match="verified complete epoch80|registered|registry"):
        prepare_parameter_package(changed, tmp_path / "missing.json", output=tmp_path / "package")


def measurement_fixture(folder):
    bindings = {"identity_sha256": "a" * 64, "clock_hz": 100000000,
                "measurement_protocol_sha256": "b" * 64, "bitstream_sha256": "c" * 64}
    latency_protocol = {"scope": "end_to_end_input_to_host_detections", "unit": "us", "batch_size": 1,
                        "warmup": 2, "repetitions": 30, "statistic": "p95", "limit_us": 2000.}
    latency = {"bindings": bindings, "evidence_kind": "synthetic_diagnostic", "scope": latency_protocol["scope"],
               "unit": "us", "batch_size": 1, "concurrency": 1, "timebase": "monotonic_ns", "warmup_discarded": 2,
               "samples": [{"repeat": index, "start_ns": index * 2000000, "end_ns": index * 2000000 + 1000000,
                            "duration_us": 1000.} for index in range(30)]}
    power_protocol = {"scope": "board_input_total", "rails": ["board_input"], "unit": "W", "time_unit": "s",
                      "measurement_device": "synthetic_fixture", "integration": "trapezoid", "idle_subtraction": "none",
                      "window_start_s": 0., "window_end_s": 2., "minimum_samples": 10, "max_sample_gap_s": .11,
                      "completed_inferences": 1, "max_average_power_w": 6., "max_energy_per_inference_j": 11.}
    power = {"bindings": bindings, "evidence_kind": "synthetic_diagnostic", "scope": "board_input_total",
             "rails": ["board_input"], "unit": "W", "time_unit": "s", "measurement_device": "synthetic_fixture",
             "timestamp_origin": "capture_start", "samples": [{"time_s": index / 10, "power_w": 5.} for index in range(21)],
             "inference_events": [{"inference": 0, "start_s": .1, "end_s": 1.9}]}
    receipt = {"measurements": {"latency": save(folder / "latency.json", latency), "power": save(folder / "power.json", power)}}
    return receipt, bindings, {"latency": latency_protocol, "power": power_protocol}


def test_raw_measurements_recomputed_without_diagnostic_promotion(tmp_path):
    receipt, bindings, protocol = measurement_fixture(tmp_path)
    result = evaluate_measurements(receipt, bindings, protocol)
    assert result["latency"]["status"] == result["power"]["status"] == "ACCEPTANCE_PASS"
    assert result["latency"]["statistics"]["p95"] == 1000.
    assert result["power"]["reported_energy_j"] == pytest.approx(10.)
    assert result["power"]["reported_average_power_w"] == pytest.approx(5.)
    assert not any(row["formal_eligible"] for row in result.values())
    assert result["accuracy"]["status"] == "NOT_VERIFIED"


@pytest.mark.parametrize("fault,stage,reason", [
    ("identity_swap", "latency", "binding differs"),
    ("too_few_repeats", "latency", "count/order"),
    ("unit_swap", "power", "units"),
    ("time_window", "power", "exact frozen time window"),
])
def test_measurement_identity_units_sample_count_and_window(tmp_path, fault, stage, reason):
    receipt, bindings, protocol = measurement_fixture(tmp_path)
    path = Path(receipt["measurements"][stage]["path"])
    raw = json.loads(path.read_text())
    if fault == "identity_swap":
        raw["bindings"]["bitstream_sha256"] = "f" * 64
    elif fault == "too_few_repeats":
        raw["samples"].pop()
    elif fault == "unit_swap":
        raw["unit"] = "mW"
    elif fault == "time_window":
        raw["samples"][-1]["time_s"] = 1.99
    receipt["measurements"][stage] = save(path, raw)
    result = evaluate_measurements(receipt, bindings, protocol)
    assert result[stage]["status"] == "BLOCKED_INVALID_EVIDENCE" and reason in result[stage]["reason"]
    assert not result[stage]["formal_eligible"]


def test_frozen_gt_cannot_be_replaced_by_self_consistent_synthetic_hashes():
    manifest = Path(__file__).resolve().parents[1] / (
        "results/space192_v2_20261013/stage_B/collected_terminal_20261013T142134Z_9cbac346/"
        "snapshot/remote/execution_manifest.json")
    if not manifest.is_file():
        pytest.skip("preserved execution manifest is unavailable")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    targets = [{"image": "valid/images/" + row["image"], "target": row["target"]} for row in payload["validation_targets"]]
    protocol = {"validation_count": 99, "validation_images_sha256": digest(sorted(row["image"] for row in targets)),
                "validation_targets_sha256": digest(sorted(targets, key=lambda row: row["image"]))}
    checkpoint = {"binding": {"execution_manifest_sha256": sha256_file(manifest),
                              "validation_target_sha256": payload["validation_target_sha256"]}}
    assert validate_validation_contract(protocol, manifest, checkpoint)["execution_manifest_sha256"] == sha256_file(manifest)
    altered = deepcopy(targets)
    altered[0]["target"] = {"boxes": [[1., 1., 10., 10.]], "labels": [0]}
    protocol["validation_targets_sha256"] = digest(sorted(altered, key=lambda row: row["image"]))
    with pytest.raises(ValueError, match="GT/image hashes differ"):
        validate_validation_contract(protocol, manifest, checkpoint)


def test_self_signed_replay_report_cannot_register_formal_evaluation(tmp_path):
    identity, receipts, _ = confirmation_fixture(tmp_path / "paired")
    receipt = json.loads(Path(receipts[0]["path"]).read_text())
    item = receipt["profiles"]["W8A8"]
    item["independent_replay"] = save(tmp_path / "self_signed_replay.json", {"status": "PASS_REAL_REPLAY"})
    receipt["execution_manifest"] = save(tmp_path / "self_signed_execution.json", {"synthetic": True})
    with pytest.raises(ValueError, match="independently reviewed evaluation registry"):
        require_registered_evaluation(identity, receipt, item)


def test_board_measured_label_and_self_signed_review_cannot_admit_capture(tmp_path):
    identity, _ = identity_fixture(tmp_path / "candidate")
    export = export_fixture(tmp_path / "export", identity)
    package = prepare_parameter_package(identity, export, output=tmp_path / "package")
    loader_path = Path(package["loader_manifest"]["path"])
    board = board_fixture(tmp_path / "board", identity, loader_path)
    receipt = json.loads(board.read_text())
    receipt["evidence_kind"] = "board_measured"
    for field in ("implementation_manifest", "loader_capture"):
        path = Path(receipt[field]["path"])
        data = json.loads(path.read_text())
        data["evidence_kind"] = "board_measured"
        receipt[field] = save(path, data)
    receipt["independent_review"] = save(tmp_path / "self_signed_board_review.json", {"status": "PASS_CAPTURE_REVIEW"})
    save(board, receipt)
    result = validate_board_return(identity, loader_path, board)
    assert result["formal_gate"] == "BLOCKED" and not result["hardware_verified"]
    assert "BOARD_CAPTURE_NOT_IN_INDEPENDENTLY_REVIEWED_REGISTRY" in result["formal_blockers"]
