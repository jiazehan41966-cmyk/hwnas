"""Fail-closed verification of returned trained-detector FPGA evidence.

Nothing here runs a board or substitutes software outputs for returned bytes.
Only a complete, route-bound train8/valid99 return can create board.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import zipfile

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools/sonar_analysis"), str(ROOT / "scripts")]
from tools.sonar_deploy import admission as gate


def _under(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("unsafe evidence path")
    return path


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _check_hash(path, expected):
    if gate.sha256(path) != expected:
        raise ValueError("artifact SHA mismatch: " + str(path))


def _raw_heads(evidence, index, graph, expected_heads, metadata):
    """Read real returned HWC bytes; independently verify transport padding."""
    if type(index) is not int or index < 0:
        raise ValueError("invalid frame index")
    rows = {r["level"]: r for r in metadata}
    if len(rows) != len(metadata) or set(rows) != set(graph["outputs"]) or set(expected_heads) != set(rows):
        raise ValueError("head levels missing/duplicated")
    descriptors = {d["name"]: d for d in graph["descriptors"]}
    values = {}
    for level, node in graph["outputs"].items():
        c, h, w = graph["edges"][node]["shape"]
        descriptor = descriptors[level]
        if [c, h, w] != [descriptor["channels"], descriptor["height"], descriptor["width"]]:
            raise ValueError("head shape/descriptor mismatch")
        size = c * h * w
        if descriptor["real_bytes"] != size or descriptor["padded_bytes"] != ((size + 31) // 32) * 32:
            raise ValueError("head output size/padding contract mismatch")
        path = _under(evidence, f"frames/{index}/{level}.bin")
        data = path.read_bytes(); digest = _hash_bytes(data)
        if len(data) != size or rows[level]["bytes"] != size:
            raise ValueError("returned head shape/length mismatch")
        if digest != expected_heads[level] or digest != rows[level]["sha256"]:
            raise ValueError("returned head SHA mismatch")
        padded = path.with_name(level + ".padded.bin").read_bytes()
        if (len(padded) != descriptor["padded_bytes"] or padded[:size] != data or
                any(padded[size:]) or rows[level].get("padding_zero") is not True):
            raise ValueError("returned head padding is not verified zero")
        values[level] = np.frombuffer(data, dtype=np.int8).reshape(h, w, c).transpose(2, 0, 1).copy()
    return values


def validate_actual_frames(evidence, graph, expected_frames, source):
    """Verify every actual trace/head against the previously bound references."""
    evidence = Path(evidence)
    rows = gate.read_json(evidence / "frames.json")
    expected = expected_frames["frames"]
    if len(rows) != 107 or len(expected) != 107:
        raise ValueError("missing/extra frames: requires train8 and valid99")
    if [r["split"] for r in expected] != ["train"] * 8 + ["valid"] * 99:
        raise ValueError("expected frames must preserve train8 then valid99 order")
    if len({r["sample_id"] for r in rows}) != 107:
        raise ValueError("duplicate returned sample identity")
    if [r["sample_id"] for r in expected[:8]] != expected_frames["calibration_first8_sample_ids"]:
        raise ValueError("train prefix changed")
    completion = gate.read_json(evidence / "completion.json")
    status = completion.get("frame_status", {})
    if (completion.get("status") != "FPGA_GRAPH_PARITY_PASS" or completion.get("real_weights_bound") is not True or
            completion.get("test_evaluated") is not False or completion.get("nodes") != len(source["layers"]) or
            status != {"train_frames": 8, "valid_frames": 99, "mismatch_count": 0, "test_evaluated": False}):
        raise ValueError("board full-detector completion evidence is incomplete")
    node_names = [row["node"] for row in source["layers"]]
    for index, (actual, planned) in enumerate(zip(rows, expected)):
        if any(actual.get(k) != v for k, v in {"index": index, "sample_id": planned["sample_id"], "split": planned["split"], "mismatch_count": 0}.items()):
            raise ValueError("returned frame identity/order/parity mismatch")
        if actual["split"] == "train":
            trace = actual.get("layer_trace", [])
            if [r["node"] for r in trace] != node_names or set(planned["expected_layers"]) != set(node_names):
                raise ValueError("train frame lacks complete ordered layer trace")
            for row in trace:
                if row.get("mismatch_count") != 0 or row["sha256"] != planned["expected_layers"][row["node"]]:
                    raise ValueError("train layer trace parity mismatch")
        _raw_heads(evidence, index, graph, planned["expected_heads"], actual["heads"])
    initial = gate.read_json(evidence / "layers.json")
    if len(initial) != len(source["layers"]):
        raise ValueError("initial full-graph witness missing nodes")
    for index, (row, expected) in enumerate(zip(initial, source["layers"])):
        if row.get("index") != index or row["node"] != expected["node"] or row.get("mismatch_count") != 0 or row["output_sha256"] != expected["output_sha256"]:
            raise ValueError("initial full-graph layer witness mismatch")
    return rows


def _validate_executor(board, source):
    executor = board.get("executor", {})
    if not executor or executor != source.get("implementation"):
        raise ValueError("board executor differs from bound build implementation")
    version = executor.get("executor_version")
    if version not in ("serial_ddr_graph_v2_requant", "serial_ddr_graph_v2_requant_timing02"):
        raise ValueError("board image lacks explicit requant support")
    if version == "serial_ddr_graph_v2_requant_timing02" and executor.get("requant_pipeline") != "off":
        raise ValueError("timing02 must bind serial requant scheduling")


def _validate_payload(build, evidence, frame_bundle, package, route, result):
    """Bind returned bitstream/HWH, ABI, routes, weights, graph, and frames."""
    from tools.detector_preflight.build_graph_witness import validate_package
    graph, export_binding = validate_package(package)
    graph_hash = gate.sha256(package / "graph.json")
    source = gate.read_json(build / "manifest.json")
    returned = gate.read_json(evidence / "source_manifest.json")
    if returned != source or source.get("real_weights_bound") is not True or source["graph_sha256"] != graph_hash:
        raise ValueError("board/build/real parameter graph mismatch")
    for name, digest in source["files"].items():
        _check_hash(_under(build, name), digest)
    board = gate.read_json(evidence / "board_manifest.json")
    prepared = build / "board_payload" / "board_manifest.json"
    if gate.sha256(prepared) != gate.sha256(evidence / "board_manifest.json"):
        raise ValueError("returned board manifest differs from packaged image")
    if board.get("test_evaluated") is not False or board.get("real_weights_bound") is not True or board.get("graph_sha256") != graph_hash:
        raise ValueError("board manifest binding/test seal failure")
    _validate_executor(board, source)
    for name, digest in board["files"].items():
        _check_hash(_under(build / "board_payload", name), digest)
    for name in ("h2_graph.bit", "h2_graph.hwh", "source_manifest.json", "graph.json", "export_manifest.json", "timing.rpt"):
        _check_hash(evidence / name, board["files"][name])
    if gate.sha256(evidence / "graph.json") != graph_hash or gate.sha256(evidence / "export_manifest.json") != gate.sha256(package / "export_manifest.json"):
        raise ValueError("returned parameter/export binding changed")
    if source.get("export_binding", {}).get("provenance") != graph["provenance"]:
        raise ValueError("build provenance differs from real package")
    with zipfile.ZipFile(build / "h2_graph.xsa") as archive:
        for name, filename in (("h2_graph.bit", "h2_graph.bit"), ("h2_graph_bd.hwh", "h2_graph.hwh")):
            if archive.namelist().count(name) != 1 or _hash_bytes(archive.read(name)) != gate.sha256(evidence / filename):
                raise ValueError("board bitstream/HWH not from routed XSA")
    headers = list((build / "hls_project/solution1/impl/ip/drivers").rglob("*_hw.h"))
    headers = [p for p in headers if gate.sha256(p) == board["header_sha256"]]
    if len(headers) != 1:
        raise ValueError("unique bound HLS ABI header unavailable")
    macros = dict(re.findall(r"#define\s+XH2_GRAPH_CONTROL_ADDR_(\w+)\s+(0x[0-9a-fA-F]+)", headers[0].read_text()))
    abi = {name: int(macros[key], 16) for name, key in {"arena": "ARENA_DATA", "weights": "WEIGHTS_DATA", "biases": "BIASES_DATA", "descriptor": "DESCRIPTOR_DATA", "return": "AP_RETURN"}.items()}
    if board["abi"] != abi:
        raise ValueError("board ABI differs from synthesized header")
    if board["implemented_clock_mhz"] != route["actual_clock_mhz"] or route["timing_met"] is not True:
        raise ValueError("board clock lacks matching routed timing")
    frames = gate.read_json(frame_bundle / "frames.json")
    if gate.sha256(frame_bundle / "frames.json") != board["files"].get("frames.json"):
        raise ValueError("board frame set differs from frozen software references")
    if (frames.get("test_evaluated") is not False or frames["graph_sha256"] != graph_hash or
            frames["package_manifest_sha256"] != result["package_manifest_sha256"] or
            frames["calibration_manifest_sha256"] != result["calibration_sha256"] or
            frames["valid_split_manifest_sha256"] != result["split_manifest_sha256"]):
        raise ValueError("frame dataset/calibration/package binding mismatch")
    for name, digest in frames["files"].items():
        _check_hash(_under(frame_bundle, name), digest)
    return graph, source, board, frames


class ReplayBoardModel(nn.Module):
    """CPU dequantization of authenticated, actual valid99 FPGA raw heads."""
    def __init__(self, graph, actual, expected, evidence, template):
        super().__init__()
        self.graph, self.evidence = graph, Path(evidence)
        self.actual = [r for r in actual if r["split"] == "valid"]
        self.expected = [r for r in expected if r["split"] == "valid"]
        if len(self.actual) != 99 or len(self.expected) != 99:
            raise ValueError("ReplayBoardModel requires exactly valid99")
        object.__setattr__(self, "_template", template)
        self.cursor = 0

    def loss_config(self, config):
        return self._template.loss_config(config)

    def forward(self, images):
        if images.device.type != "cpu" or images.ndim != 4 or tuple(images.shape[1:]) != (1, 640, 640):
            raise ValueError("board replay uses frozen CPU decode with NCHW640")
        count = images.shape[0]
        if self.cursor + count > 99:
            raise ValueError("board replay consumed too many valid samples")
        batched = {level: [] for level in self.graph["outputs"]}
        for offset in range(count):
            i = self.cursor + offset
            actual, planned = self.actual[i], self.expected[i]
            if actual["sample_id"] != planned["sample_id"]:
                raise ValueError("board replay sample order changed")
            values = _raw_heads(self.evidence, actual["index"], self.graph, planned["expected_heads"], actual["heads"])
            for level, node in self.graph["outputs"].items():
                batched[level].append(torch.from_numpy(values[level]).to(torch.float32) * (2.0 ** self.graph["edges"][node]["exp"]))
        self.cursor += count
        return {"predictions": tuple(torch.stack(batched[level]) for level in self.graph["outputs"])}


def evaluate_board_cpu(root, result, graph, actual, frames, evidence):
    import baseline as frozen
    from phase5a_quantization.common import baseline, build_network
    from hwnas_fpga.training.detection_trainer import DenseAnchorDetectionLoss
    template, aid = build_network(result["network"])
    if aid != result["architecture_id"]:
        raise ValueError("board evaluation architecture changed")
    config = baseline()
    loader = frozen.make_loader(config, frozen.resolve(config["dataset"]["data_dir"]), "valid", 42, 0)
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError("CPU board replay needs deterministic valid order")
    bound = frozen.bind_dataset(loader)
    pinned = gate.read_json(Path(root) / "VALID_MANIFEST.json")
    if bound["entries"] != pinned["entries"] or bound["images"] != 99:
        raise ValueError("CPU board evaluation dataset identity changed")
    expected = [r for r in frames["frames"] if r["split"] == "valid"]
    if [r["sample_id"] for r in expected] != [r["image"].replace("\\", "/") for r in bound["entries"]]:
        raise ValueError("actual board frame order differs from frozen valid99")
    model = ReplayBoardModel(graph, actual, frames["frames"], evidence, template).eval()
    criterion = DenseAnchorDetectionLoss(template.loss_config(config)).cpu()
    metrics, records = frozen.evaluate(model, loader, criterion, config, "cpu")
    if model.cursor != 99 or metrics["num_images"] != 99:
        raise ValueError("CPU board evaluation did not consume all valid99")
    return metrics, records


def _timing_from_e2e(path, board, graph_hash, completion):
    data = gate.read_json(path)
    required = {"schema_version": "real-int8-e2e-v1", "status": "PASS", "precision": "INT8", "quantization_policy": "W8A8",
                "test_evaluated": False, "batch_size": 1, "input_size": 640, "warmup": 1, "repetitions": 10,
                "weights_resident": True, "timing_scope": "host_preprocess_rpc_board_graph_receive_cpu_decode"}
    if any(data.get(k) != v for k, v in required.items()):
        raise ValueError("end-to-end timing protocol differs")
    if data.get("graph_sha256") != graph_hash or data.get("board_manifest_sha256") != gate.sha256(Path(path).parent / "board_manifest.json"):
        raise ValueError("end-to-end timing not bound to this board image")
    samples = data.get("samples")
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("end-to-end requires ten actual measured samples")
    times = {"graph_seconds": [], "end_to_end_seconds": []}
    if (len(data.get("rows", [])) != 11 or data["rows"][0].get("warmup") is not True or
            data["rows"][0].get("call_index") != 0 or data["rows"][1:] != samples):
        raise ValueError("benchmark warmup/sample record mismatch")
    for i, sample in enumerate(samples, 1):
        if sample.get("call_index") != i or sample.get("warmup") is not False:
            raise ValueError("benchmark measured call ordering differs")
        for key in times:
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid board timing sample")
            times[key].append(value)
        if sample["end_to_end_seconds"] < sample["graph_seconds"]:
            raise ValueError("end-to-end timer excludes graph execution")
        sample_clock = sample.get("board_clock_mhz")
        if (isinstance(sample_clock, bool) or not isinstance(sample_clock, (int, float)) or
                not math.isfinite(sample_clock) or abs(sample_clock - board["implemented_clock_mhz"]) > .1):
            raise ValueError("sample clock differs from routed image")
    actual_clock = completion.get("board_clock_mhz")
    if not isinstance(actual_clock, (int, float)) or not math.isfinite(actual_clock) or abs(actual_clock - board["implemented_clock_mhz"]) > .1:
        raise ValueError("board runtime clock differs from route")
    for scope in ("graph", "end_to_end"):
        values = times[scope + "_seconds"]
        times[scope + "_median_ms"] = statistics.median(values) * 1000
        times[scope + "_p95_ms"] = float(np.percentile(values, 95)) * 1000
    for key, value in times.items():
        recorded = data.get("timing", {}).get(key)
        if isinstance(value, list):
            if recorded != value: raise ValueError("timing raw sample arrays disagree")
        elif not isinstance(recorded, (int, float)) or not math.isclose(recorded, value, rel_tol=1e-12):
            raise ValueError("timing aggregate disagrees with raw measurements")
    return times, actual_clock, data


def _validate_rpc_session(path, e2e):
    """Require board-side completion after the last response and buffer release.

    A host can receive the eleventh response before GraphSession.close raises.
    Its timings alone therefore cannot prove the board session ended safely.
    """
    rpc = gate.read_json(path)
    if (rpc.get("status") != "PASS" or type(rpc.get("successful_calls")) is not int or
            rpc["successful_calls"] != 11 or rpc.get("weights_resident") is not True or
            rpc.get("test_evaluated") is not False):
        raise ValueError("board RPC session did not complete eleven calls and release buffers safely")
    for key in ("graph_sha256", "package_manifest_sha256", "board_manifest_sha256"):
        if not e2e.get(key) or rpc.get(key) != e2e[key]:
            raise ValueError("board RPC session identity differs: " + key)
    rows, host_rows = rpc.get("rows"), e2e.get("rows")
    if not isinstance(rows, list) or len(rows) != 11 or not isinstance(host_rows, list) or len(host_rows) != 11:
        raise ValueError("board RPC session requires eleven actual call records")
    for index, (board_row, host_row) in enumerate(zip(rows, host_rows)):
        if (type(board_row.get("call_index")) is not int or board_row["call_index"] != index or
                host_row.get("call_index") != index or board_row.get("test_evaluated") is not False):
            raise ValueError("board RPC call order/test seal differs")
        for key in ("graph_seconds", "input_graph_capture_seconds", "board_clock_mhz"):
            value = board_row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0 or value != host_row.get(key)):
                raise ValueError("board RPC measurement differs from host response: " + key)
        if board_row["input_graph_capture_seconds"] < board_row["graph_seconds"]:
            raise ValueError("board RPC timing excludes graph execution")
    return rpc


def verify_board_result(root, network):
    """Verify local returned bytes, evaluate CPU, then publish durable board PASS."""
    root = Path(root).resolve()
    if network not in gate.NETWORKS:
        raise ValueError("only frozen B1/C/D")
    folder = root / network / "seed_42"; build = folder / "hardware/graph_build"
    candidates = [p for p in (folder / "hardware/board_evidence", build / "board_evidence") if (p / "completion.json").is_file()]
    if not candidates:
        return dict(status="WAITING_FOR_LOCAL_BOARD_RESULT", network=network, allowed=False, test_evaluated=False)
    if len(candidates) != 1:
        raise ValueError("ambiguous board return; explicit evidence selection required")
    evidence = candidates[0]
    if not (evidence / "e2e.json").is_file():
        return dict(status="WAITING_FOR_LOCAL_BOARD_RESULT", network=network, allowed=False,
                    reason="actual ten-sample host-to-board end-to-end timing missing", test_evaluated=False)
    if not (evidence / "e2e_rpc_session.json").is_file():
        return dict(status="WAITING_FOR_LOCAL_BOARD_RESULT", network=network, allowed=False,
                    reason="board RPC session completion after buffer release missing", test_evaluated=False)
    admitted = gate.action_admission(root, network, "board", 42)
    if not admitted["allowed"]:
        return admitted
    binding = gate.load_binding(root)
    row = next(r for r in binding["runs"] if r["network"] == network and r["seed"] == 42)
    result = gate.load_result(root, row)
    if (folder / "board.json").is_file():
        return gate.stage_evidence(root, network, "board", result)
    route = gate.stage_evidence(root, network, "route", result)
    package_path = Path(result["package_manifest_path"])
    package = package_path.parent if package_path.is_absolute() else (root / package_path).parent
    graph, source, board, frames = _validate_payload(build, evidence, folder / "hardware/frame_bundle", package, route, result)
    actual = validate_actual_frames(evidence, graph, frames, source)
    metrics, records = evaluate_board_cpu(root, result, graph, actual, frames, evidence)
    gate.atomic_json(evidence / "board_valid_metrics.json", metrics)
    gate.atomic_json(evidence / "board_valid_predictions.json", records)
    differences = {k: metrics[k] - result["integer_metrics"][k] for k in gate.METRICS if metrics[k] != result["integer_metrics"][k]}
    if differences:
        report = dict(status="FAIL_BOARD_VALID_METRIC_MISMATCH", reason="CPU board decode differs from frozen integer reference; no tolerance/protocol adjustment", differences=differences, test_evaluated=False)
        gate.atomic_json(evidence / "BOARD_RESULT_REVIEW.json", report)
        return report
    completion = gate.read_json(evidence / "completion.json")
    timing, clock, e2e = _timing_from_e2e(evidence / "e2e.json", board, gate.sha256(package / "graph.json"), completion)
    rpc = _validate_rpc_session(evidence / "e2e_rpc_session.json", e2e)
    if (e2e.get("deployment_binding_sha256") != gate.sha256(root / "DEPLOYMENT_BINDING.json") or
            e2e.get("package_manifest_sha256") != result["package_manifest_sha256"]):
        raise ValueError("end-to-end benchmark belongs to a different deployment")
    from tools.sonar_deploy.board_e2e import validate_manifest
    benchmark_path = evidence / "benchmark_manifest.json"
    _check_hash(benchmark_path, e2e["benchmark_manifest_sha256"])
    benchmark = validate_manifest(gate.read_json(benchmark_path))
    if benchmark["rpc_source_sha256"] != e2e.get("source_sha256"):
        raise ValueError("end-to-end RPC implementation differs from bound source")
    for key in ("deployment_binding_sha256", "package_manifest_sha256", "graph_sha256", "board_manifest_sha256"):
        if benchmark[key] != e2e[key]:
            raise ValueError("benchmark envelope/result identity mismatch")
    first = frames["frames"][0]
    if (benchmark["network"] != network or benchmark["calibration_sha256"] != result["calibration_sha256"] or
            benchmark["sample"].get("calibration_index") != 0 or benchmark["sample"].get("image") != first["sample_id"] or
            {h["level"]: h["sha256"] for h in benchmark["heads"]} != first["expected_heads"]):
        raise ValueError("timing witness differs from first frozen train image")
    input_value = np.load(folder / "hardware/frame_bundle" / first["input"], allow_pickle=False)
    if _hash_bytes(input_value.copy(order="C").tobytes()) != benchmark["input_sha256"]:
        raise ValueError("timing preprocessing input differs from integer witness")
    if e2e.get("mismatch_count") != 0 or e2e.get("integer_witness_sample_count") != 1 or e2e.get("expected_head_sha256") != first["expected_heads"]:
        raise ValueError("end-to-end runs lack matching raw-head witness")
    power_result = dict(status="NOT_MEASURED", reason="optional sensor capture not enabled")
    power_config = benchmark.get("power_monitor", {})
    if power_config.get("enabled") is True:
        from tools.sonar_deploy.power import verify_iio_evidence
        try:
            power_binding = {key: e2e[key] for key in ("deployment_binding_sha256", "package_manifest_sha256",
                "board_manifest_sha256", "graph_sha256", "benchmark_manifest_sha256")}
            power_result = verify_iio_evidence(evidence / "power", power_binding,
                                               power_config["source_sha256"], rpc["rows"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            power_result = dict(status="NOT_MEASURED", reason="INVALID_OR_INCOMPLETE_POWER_EVIDENCE: " + str(exc))
    artifacts = {"returned_" + p.relative_to(evidence).as_posix(): p for p in evidence.rglob("*") if p.is_file() and p.name != "BOARD_RESULT_REVIEW.json"}
    artifacts.update(expected_frames=folder / "hardware/frame_bundle/frames.json", build_manifest=build / "manifest.json",
                     graph=package / "graph.json", export_manifest=package / "export_manifest.json")
    provenance = dict(source=str(evidence / "e2e.json"), artifact_sha256=gate.sha256(evidence / "e2e.json"),
        evidence_layer="board_measurement", target_device="ZCU104", measurement_method="host_monotonic_preprocess_rpc_board_graph_receive_cpu_decode",
        precision="W8A8", batch_size=1, input_shape=[1, 1, 640, 640], warmup=1, repetitions=10,
        checkpoint_sha256=result["checkpoint_sha256"], evaluation_protocol_hash=result["evaluator_sha256"],
        split_manifest_sha256=result["split_manifest_sha256"], graph_sha256=gate.sha256(package / "graph.json"))
    values = {"board_latency_ms": timing["end_to_end_median_ms"], "throughput_fps": 1000 / timing["end_to_end_median_ms"],
              "actual_clock_mhz": clock, **{target: metrics[source] for source, target in gate.METRICS.items()}}
    measured = {key: dict(value=value, provenance=provenance) for key, value in values.items()}
    if power_result["status"] == "MEASURED":
        power_provenance = dict(provenance, source=str(evidence / "power/power.json"),
            artifact_sha256=gate.sha256(evidence / "power/power.json"),
            measurement_method="board_monotonic_IIO_raw_scale_trapezoidal_integration_per_infer_call",
            power_scope=power_result["scope"], idle_subtracted=False,
            energy_scope=power_result["summary"]["energy_scope"],
            repetitions=10)
        for name in ("power_w", "energy_per_inference_mj"):
            measured[name] = dict(value=power_result["summary"][name], provenance=power_provenance)
    from tools.sonar_deploy.hardware import _publish
    return _publish(root, network, "board", result, artifacts,
        dict(mismatch_count=0, train_sample_count=8, valid_sample_count=99, warmup=1, repetitions=10, batch_size=1,
             valid_metrics=metrics, timing=timing, actual_clock_mhz=clock, power_status=power_result["status"],
             power_evidence=power_result,
             cpu_decode_reference="frozen evaluator exact five-metric match", test_accessed=False), measured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--network", choices=gate.NETWORKS, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    print(json.dumps(verify_board_result(args.root, args.network), indent=2))


if __name__ == "__main__":
    main()
