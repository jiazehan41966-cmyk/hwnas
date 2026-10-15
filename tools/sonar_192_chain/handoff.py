"""Integer transport packaging and source-bound full-graph board-return checks.

The transport ABI is a software contract, not a claim that an existing FPGA
bitstream implements it. No function programs a board or launches hardware work.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from math import prod
from pathlib import Path

import numpy as np

from .contract import digest, sha256_file, validate_identity
from .confirmation import (
    DEPLOYMENT_SEED, METRIC, artifact, read_json, reference, replay_confirmation, require, score_records, validate_validation_contract,
)

ABI = "sonar192_integer_transport_v1"
PACKAGE_SCHEMA = "sonar192_parameter_package_v1"
BOARD_SCHEMA = "sonar192_board_return_v1"
MEASUREMENT_SCHEMA = "sonar192_measurement_protocol_v1"
# Reviewed board-capture artifacts, not self-asserted measurement labels. No
# runner/API/CLI can enroll an entry. There are no eligible board captures yet.
REGISTERED_FORMAL_BOARD_RETURNS = ()
EXPORT_FILES = ("profile.json", "numeric_graph.json", "integer_parameters.npz", "parameter_index.json",
                "software_parity_witness.json", "reference_outputs.npz", "reference_index.json")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _calibration_hash(value):
    require(isinstance(value, dict) and value.get("calibration_sha256", value.get("sha256")), "Missing calibration identity")
    return value.get("calibration_sha256", value.get("sha256"))


def _verify_calibration_sources(calibration, candidate):
    if "calibration_sha256" not in calibration:
        require(candidate["purpose"] == "diagnostic", "Formal calibration must carry source manifest")
        return
    body = deepcopy(calibration)
    checksum = body.pop("calibration_sha256")
    require(digest(body) == checksum, "Calibration manifest checksum differs")
    require(body.get("split") == "train" and all(body.get(key) is False for key in
            ("augmentation", "shuffle", "valid_used_for_selection", "test_accessed", "test_evaluated")),
            "Calibration must preserve train-only unaugmented data")
    manifest_path = Path(body["training_execution_manifest_path"])
    require(sha256_file(manifest_path) == body["training_execution_manifest_sha256"] ==
            candidate["checkpoint"]["binding"]["execution_manifest_sha256"], "Calibration execution source differs from EMA run")
    manifest = _json(manifest_path)
    train = manifest["inputs"]["datasets"]["train"]
    require(digest(train["entries"]) == train["sha256"] == body["frozen_train_entries_sha256"],
            "Frozen calibration train inventory differs")
    count = body["number_of_calibration_images"]
    require(type(count) is int and count > 0 and body["entries"] == train["entries"][:count],
            "Calibration is not the frozen sorted training prefix")
    if candidate["purpose"] == "formal":
        require(count == 64 and body.get("formal_calibration_count") is True, "Formal calibration requires train64")
    dataset_root = Path(body["dataset_root"]).resolve()
    for row in body["entries"]:
        for field, kind in (("image", "images"), ("annotation", "labels")):
            relative = Path(row[field].replace("\\", "/"))
            require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[:2] == ("train", kind),
                    "Calibration source is not train-only")
            source = (dataset_root / relative).resolve()
            require(source.is_relative_to(dataset_root) and sha256_file(source) == row[field + "_sha256"],
                    "Calibration image/annotation source drift")


def inspect_export(candidate, manifest_path):
    """Verify every saved file, complete integer tensors, and raw node witnesses."""
    validate_identity(candidate)
    manifest_path = Path(manifest_path).resolve()
    manifest = _json(manifest_path)
    files = manifest["files"]
    require(set(EXPORT_FILES) <= set(files), "PTQ export lacks transport/raw witness files")
    for name, expected in files.items():
        require(Path(name).name == name, "PTQ manifest must not escape its export directory")
        require(sha256_file(manifest_path.parent / name) == expected, "PTQ file SHA256 mismatch: " + name)
    folder = manifest_path.parent
    profile, graph, witness, index, ref_index = [_json(folder / name) for name in
        ("profile.json", "numeric_graph.json", "software_parity_witness.json", "parameter_index.json", "reference_index.json")]
    for key in ("space_version", "architecture_sha256", "full_contract_sha256"):
        require(profile[key] == candidate[key], "PTQ identity mismatch: " + key)
    require(profile["precision_profile"] == candidate["profile"], "PTQ precision differs")
    require(profile["checkpoint_binding"] == candidate["checkpoint"], "PTQ checkpoint binding differs")
    require(profile["original_model_state_sha256"] == candidate["checkpoint"]["model_state_sha256"],
            "PTQ original model weights differ from selected EMA")
    require(_calibration_hash(profile["calibration"]) == _calibration_hash(candidate["calibration"]),
            "PTQ calibration differs")
    _verify_calibration_sources(profile["calibration"], candidate)
    if "identity_sha256" in profile:
        require(profile["identity_sha256"] == candidate["identity_sha256"], "PTQ full identity differs")
    from .quantization import numeric_contract
    registered_numeric, registered_numeric_hash = numeric_contract(candidate["profile"])
    require(profile["numeric_contract"] == registered_numeric and profile["numeric_contract_hash"] == registered_numeric_hash,
            "Unregistered integer arithmetic")
    require(candidate["mapping"]["numeric_contract_sha256"] == registered_numeric_hash,
            "Mapping numeric contract differs from quantization")
    require(profile.get("source_sha256"), "Missing quantization source binding")
    if candidate["purpose"] == "formal":
        root = Path(__file__).resolve().parents[2]
        required_sources = {str((root / item).resolve()) for item in
                            ("tools/sonar_192_chain/quantization.py", "tools/sonar_192_chain/graph.py", "tools/fpga_h2/numeric.py")}
        require(required_sources <= set(profile["source_sha256"]), "Formal PTQ missing actual arithmetic/graph source hashes")
    for source, checksum in profile["source_sha256"].items():
        require(sha256_file(source) == checksum, "Quantization source changed: " + source)
    graph_hash = graph["numeric_graph_sha256"]
    graph_body = deepcopy(graph)
    del graph_body["numeric_graph_sha256"]
    require(digest(graph_body) == graph_hash == profile["numeric_graph_sha256"], "Numeric graph hash differs")
    for key in ("space_version", "architecture_sha256", "full_contract_sha256"):
        require(graph[key] == candidate[key], "Graph model/head identity differs: " + key)
    nodes = {node["id"]: node for node in graph["nodes"]}
    require(len(nodes) == len(graph["nodes"]) and nodes, "Duplicate or empty graph nodes")
    available = {"image"}
    for node in graph["nodes"]:
        require(node["inputs"] and all(name in available for name in node["inputs"]), "Invalid topological order")
        require(node["id"] not in available, "Duplicated graph/input node")
        available.add(node["id"])
    require(set(graph["outputs"]) == {"P2", "P3", "P4", "P5"}
            and all(name in nodes for name in graph["outputs"].values()), "Four-scale output contract differs")
    if candidate["purpose"] == "formal":
        from .model import build_model
        from .graph import build_graph
        model = build_model(candidate["architecture"], candidate["checkpoint"]["seed"])
        registered = build_graph(model)
        del model
        require(registered["structural_graph_sha256"] == profile["structural_graph_sha256"]
                == graph["structural_graph_sha256"], "Formal topology differs from actual frozen model")
        boundaries = {node["id"]: node for node in graph["nodes"] if node["op"] == "requantize"}
        expected_boundaries = set() if candidate["profile"] == "W8A8" else {
            "head.input_requant." + level for level in ("P2", "P3", "P4", "P5")}
        require(set(boundaries) == expected_boundaries, "Mixed head boundary coverage differs")
        actual_nodes = [deepcopy(node) for node in graph["nodes"] if node["id"] not in boundaries]
        for node in actual_nodes:
            node["inputs"] = [boundaries[name]["inputs"][0] if name in boundaries else name for name in node["inputs"]]
        require(registered["outputs"] == graph["outputs"] and registered["input_shape"] == graph["input_shape"]
                and len(registered["nodes"]) == len(actual_nodes), "Formal graph coverage differs")
        for expected_node, actual_node in zip(registered["nodes"], actual_nodes):
            for field in ("id", "op", "inputs", "input_shapes", "output_shape", "Cin", "Cout", "kernel",
                          "stride", "padding", "groups", "activation", "fixed_kernel", "denominator", "channel_order"):
                require(expected_node.get(field) == actual_node.get(field), "Formal node topology differs: " + field)
        for index, level in enumerate(("P2", "P3", "P4", "P5")):
            if not boundaries:
                break
            boundary_id = "head.input_requant." + level
            boundary = boundaries[boundary_id]
            consumers = [node for node in graph["nodes"] if boundary_id in node["inputs"]]
            require({node["id"] for node in consumers} == {f"head.cv2.{index}.0.conv", f"head.cv3.{index}.0.conv"}
                    and boundary.get("input_bits") == 8 and boundary.get("output_bits") == 16
                    and len(boundary["inputs"]) == 1
                    and boundary["output_shape"] == boundary["input_shapes"][0], "Mixed input conversion differs")
    arrays, parameter_hashes, lut_hashes = {}, {}, {}
    with np.load(folder / "integer_parameters.npz", allow_pickle=False) as payload:
        require(set(payload.files) == {item["key"] for item in index["parameters"]}, "Parameter index coverage differs")
        require(len(index["parameters"]) == len(payload.files), "Duplicate parameter index key")
        for item in index["parameters"]:
            require(item["node"] in nodes and item["kind"] in ("weight", "bias", "lut"), "Unknown parameter node/kind")
            array = payload[item["key"]]
            require(array.dtype.kind == "i" and array.dtype.itemsize in (1, 2, 4), "Parameters must be integer tensors")
            require(array.dtype == np.dtype(item["dtype"]) and list(array.shape) == item["shape"], "Parameter dtype/shape differs")
            if item["kind"] == "bias":
                require(array.dtype == np.dtype("int32"), "Bias must be INT32")
                require(list(array.shape) == [nodes[item["node"]]["Cout"]], "Bias graph geometry differs")
            elif item["kind"] == "weight":
                node = nodes[item["node"]]
                require(node.get("weight_bits", 8) in (8, 16)
                        and array.dtype == np.dtype("int" + str(node.get("weight_bits", 8))) and list(array.shape) ==
                        [node["Cout"], node["Cin"] // node["groups"], node["kernel"], node["kernel"]],
                        "Weight graph geometry/dtype differs")
            else:
                node = nodes[item["node"]]
                require(node.get("input_bits", 8) in (8, 16) and node.get("output_bits", 8) in (8, 16)
                        and array.dtype == np.dtype("int" + str(node.get("output_bits", 8)))
                        and array.shape == (2 ** node.get("input_bits", 8),), "SiLU lookup domain differs")
            require(_array_sha(array) == item["sha256"], "Integer parameter byte hash differs")
            if item["kind"] == "lut":
                lut_hashes[item["node"]] = _array_sha(array)
            else:
                parameter_hashes.setdefault(item["node"], {})[item["kind"] + "_sha256"] = _array_sha(array)
            arrays[item["key"]] = array.copy()
    require(digest({"parameters": parameter_hashes, "luts": lut_hashes}) == profile["integer_parameters_sha256"],
            "Integer parameter aggregate hash differs")
    expected_pairs = set()
    for node in graph["nodes"]:
        if node["op"] == "conv":
            expected_pairs.update(((node["id"], "weight"), (node["id"], "bias")))
        if node["op"] in ("silu", "lut", "silu_lut"):
            expected_pairs.add((node["id"], "lut"))
    pairs = [(item["node"], item["kind"]) for item in index["parameters"]]
    require(len(set(pairs)) == len(pairs) and set(pairs) == expected_pairs, "Missing/extra integer parameters")
    require(witness.get("mismatches") == 0 and witness.get("numeric_graph_sha256") == graph_hash
            and witness.get("integer_parameters_sha256") == profile["integer_parameters_sha256"]
            and witness.get("identity_sha256") == candidate["identity_sha256"]
            and witness.get("original_model_state_sha256") == candidate["checkpoint"]["model_state_sha256"],
            "Software parity witness differs")
    require(type(witness.get("input_images")) is int and witness["input_images"] > 0, "Empty software witness")
    witness_rows = {row["node"]: row for row in witness["rows"]}
    require(len(witness_rows) == len(witness["rows"]) and set(witness_rows) == available,
            "Software witness must include every graph node and input")
    require(ref_index["numeric_graph_sha256"] == graph_hash
            and ref_index["integer_parameters_sha256"] == profile["integer_parameters_sha256"],
            "Raw reference graph/parameter binding differs")
    reference_rows = {row["node"]: row for row in ref_index["nodes"]}
    require(len(reference_rows) == len(ref_index["nodes"]) and set(reference_rows) == available,
            "Raw references must include every graph node and input")
    with np.load(folder / "reference_outputs.npz", allow_pickle=False) as refs:
        require(set(refs.files) == {row["key"] for row in reference_rows.values()}, "Raw reference index differs")
        for name, row in reference_rows.items():
            value = refs[row["key"]]
            require(value.dtype.kind == "i" and value.dtype.itemsize in (1, 2)
                    and value.dtype == np.dtype(row["dtype"]) and list(value.shape) == row["shape"],
                    "Raw reference dtype/shape differs: " + name)
            expected = witness_rows[name]
            require(_array_sha(value) == row["sha256"] == expected["integer_sha256"]
                    == expected["fake_quantized_sha256"] and expected["mismatch_count"] == 0
                    and value.size == expected["element_count"], "Raw software reference mismatch: " + name)
            shape = graph["input_shape"] if name == "image" else nodes[name]["output_shape"]
            require(list(value.shape[1:]) == shape and len(value.shape) == 4
                    and value.shape[0] == witness["input_images"], "Raw reference graph geometry differs: " + name)
        if candidate["purpose"] == "formal":
            from .quantization import IntegerReference
            parameters, luts = {}, {}
            for item in index["parameters"]:
                if item["kind"] == "lut":
                    luts[item["node"]] = arrays[item["key"]]
                else:
                    parameters.setdefault(item["node"], {})[item["kind"]] = arrays[item["key"]]
            runtime = IntegerReference(graph, parameters, luts)
            require(runtime.parameter_hash() == profile["integer_parameters_sha256"], "Independent integer parameter hash differs")
            replayed = runtime.forward(refs[reference_rows["image"]["key"]])
            require(set(replayed) == set(reference_rows), "Independent integer replay node coverage differs")
            for name, value in replayed.items():
                require(np.array_equal(value, refs[reference_rows[name]["key"]]),
                        "Independent integer replay differs from saved reference: " + name)
    return {"manifest": manifest, "manifest_path": manifest_path, "profile": profile, "graph": graph,
            "witness": witness, "parameter_index": index, "arrays": arrays,
            "reference_index": ref_index, "reference_rows": reference_rows}


def _bindings(candidate, exported):
    profile = exported["profile"]
    return {"identity_sha256": candidate["identity_sha256"],
            "checkpoint_sha256": candidate["checkpoint"]["sha256"],
            "weight_sha256": candidate["checkpoint"]["model_state_sha256"],
            "graph_sha256": exported["graph"]["numeric_graph_sha256"],
            "quant_contract_sha256": profile["numeric_contract_hash"],
            "calibration_sha256": _calibration_hash(candidate["calibration"]),
            "mapping_sha256": digest(candidate["mapping"]),
            "integer_parameters_sha256": profile["integer_parameters_sha256"],
            "export_manifest_sha256": sha256_file(exported["manifest_path"])}


def _activation_layout(graph, alignment):
    """Conservative separate DDR buffers, batch1, with no hidden liveness reuse."""
    specs = [("image", graph["input_shape"], 8)] + [
        (node["id"], node["output_shape"], node.get("output_bits", 8)) for node in graph["nodes"]]
    offset, rows = 0, []
    for node, shape, bits in specs:
        require(all(type(size) is int and size > 0 for size in shape) and len(shape) == 3
                and bits in (8, 16), "Invalid activation DDR geometry/bitwidth")
        start = (offset + alignment - 1) // alignment * alignment
        nbytes = prod(shape) * bits // 8
        rows.append({"node": node, "shape_chw": shape, "bits": bits, "batch": 1,
                     "offset": start, "nbytes": nbytes, "address_space": "activations"})
        offset = start + nbytes
    return rows, offset


def prepare_parameter_package(candidate, quant_export, *, output, confirmation=None, root=".", alignment=64):
    """Write a binary, aligned DDR layout and loader manifest from real PTQ files.

    quant_export may be a path to artifact_manifest.json or a {path,sha256} ref.
    A formal-eligible package additionally needs a replayable confirmation report.
    """
    validate_identity(candidate)
    if candidate["purpose"] == "formal":
        require(candidate["checkpoint"]["seed"] == DEPLOYMENT_SEED,
                "Formal deployment package requires fixed seed42 epoch80 EMA")
    require(type(alignment) is int and alignment >= 16 and alignment <= 4096
            and alignment & (alignment - 1) == 0, "DDR alignment must be a power of two in [16,4096]")
    manifest_path = artifact(quant_export, root) if isinstance(quant_export, dict) else Path(quant_export).resolve()
    exported = inspect_export(candidate, manifest_path)
    out = Path(output).resolve()
    require(not out.exists() or not any(out.iterdir()), "Parameter output directory must be empty")
    formal, confirmation_report = False, None
    if confirmation is not None:
        confirmation_report = replay_confirmation(read_json(confirmation, root), root=root)
        require(confirmation_report["candidate_identity_sha256"] == candidate["identity_sha256"], "Confirmation candidate differs")
        formal = confirmation_report["formal_eligible"]
    out.mkdir(parents=True, exist_ok=True)
    offset, layout, blob = 0, [], bytearray()
    for item in exported["parameter_index"]["parameters"]:
        array = exported["arrays"][item["key"]]
        dtype = array.dtype.newbyteorder("<")
        raw = np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes()
        aligned = (offset + alignment - 1) // alignment * alignment
        blob.extend(b"\0" * (aligned - offset))
        blob.extend(raw)
        layout.append({**item, "dtype": dtype.str, "offset": aligned, "nbytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()})
        offset = aligned + len(raw)
    binary = out / "parameters.bin"
    binary.write_bytes(blob)
    graph = exported["graph"]
    profile = exported["profile"]
    bindings = _bindings(candidate, exported)
    ddr = {"schema_version": PACKAGE_SCHEMA, "abi": ABI, "byte_order": "little", "alignment": alignment,
           "total_bytes": len(blob), "base_address": "loader_supplied", "parameters": layout, "bindings": bindings}
    activation_buffers, activation_bytes = _activation_layout(graph, alignment)
    ddr.update(activation_buffers=activation_buffers, activation_bytes=activation_bytes,
               activation_allocation="one_nonoverlapping_buffer_per_node_batch1_no_reuse",
               address_spaces={"parameters": {"base": "loader_supplied", "bytes": len(blob)},
                               "activations": {"base": "loader_supplied_separate_region", "bytes": activation_bytes}})
    _save(out / "ddr_layout.json", ddr)
    loader = {"schema_version": PACKAGE_SCHEMA, "abi": ABI, "bindings": bindings,
              "binary": reference(binary), "ddr_layout": reference(out / "ddr_layout.json"),
              "quant_export": reference(manifest_path), "candidate": candidate,
              "confirmation": confirmation, "formal_eligible": bool(formal),
              "transport_only": True, "existing_bitstream_compatibility": "UNVERIFIED",
              "numeric_graph": reference(manifest_path.parent / "numeric_graph.json"),
              "launch_hardware": False, "node_order": [node["id"] for node in graph["nodes"]],
              "outputs": graph["outputs"], "reference_outputs": reference(manifest_path.parent / "reference_outputs.npz"),
              "reference_index": reference(manifest_path.parent / "reference_index.json")}
    _save(out / "loader_manifest.json", loader)
    report = {"schema_version": PACKAGE_SCHEMA, "status": "SOFTWARE_PARAMETER_PACKAGE_BUILT",
              "formal_eligible": bool(formal), "bindings": bindings,
              "loader_manifest": reference(out / "loader_manifest.json"), "parameter_bytes": len(blob),
              "parameters": len(layout), "activation_bytes": activation_bytes,
              "hardware_compatible": False, "launch_hardware": False,
              "limitation": "Software transport ABI; existing bitstreams have not been shown to implement it"}
    _save(out / "package_report.json", report)
    return report


def inspect_parameter_package(candidate, loader_path, *, root="."):
    validate_identity(candidate)
    if candidate["purpose"] == "formal":
        require(candidate["checkpoint"]["seed"] == DEPLOYMENT_SEED,
                "Formal deployment package requires fixed seed42 epoch80 EMA")
    loader_path = Path(loader_path).resolve()
    loader = _json(loader_path)
    require(loader["schema_version"] == PACKAGE_SCHEMA and loader["abi"] == ABI
            and loader["candidate"] == candidate, "Loader/candidate identity differs")
    exported = inspect_export(candidate, artifact(loader["quant_export"], root))
    require(loader["bindings"] == _bindings(candidate, exported), "Loader identity bindings differ")
    require(loader["node_order"] == [node["id"] for node in exported["graph"]["nodes"]]
            and loader["outputs"] == exported["graph"]["outputs"], "Loader execution order/output contract differs")
    require(read_json(loader["numeric_graph"], root) == exported["graph"], "Loader numeric graph differs")
    for field, filename in (("reference_outputs", "reference_outputs.npz"), ("reference_index", "reference_index.json")):
        require(loader[field]["sha256"] == exported["manifest"]["files"][filename], "Loader raw reference source differs")
        artifact(loader[field], root)
    binary = artifact(loader["binary"], root).read_bytes()
    ddr = read_json(loader["ddr_layout"], root)
    require(ddr["bindings"] == loader["bindings"] and ddr["total_bytes"] == len(binary), "DDR binding/length differs")
    require(ddr["abi"] == ABI and ddr["byte_order"] == "little"
            and type(ddr["alignment"]) is int and 16 <= ddr["alignment"] <= 4096
            and ddr["alignment"] & (ddr["alignment"] - 1) == 0, "Transport ABI differs")
    activation_buffers, activation_bytes = _activation_layout(exported["graph"], ddr["alignment"])
    require(ddr["activation_buffers"] == activation_buffers and ddr["activation_bytes"] == activation_bytes,
            "Activation DDR coverage/offsets/lengths differ")
    require(ddr["address_spaces"] == {"parameters": {"base": "loader_supplied", "bytes": len(binary)},
                                     "activations": {"base": "loader_supplied_separate_region", "bytes": activation_bytes}},
            "Parameter/activation DDR address spaces differ")
    index = exported["parameter_index"]["parameters"]
    require(len(ddr["parameters"]) == len(index), "DDR parameter count differs")
    previous = 0
    for row, original in zip(ddr["parameters"], index):
        for key in ("node", "kind", "key", "shape"):
            require(row[key] == original[key], "DDR index differs: " + key)
        require(np.dtype(row["dtype"]) == np.dtype(original["dtype"]).newbyteorder("<"), "DDR dtype differs")
        start, end = row["offset"], row["offset"] + row["nbytes"]
        require(start >= previous and start % ddr["alignment"] == 0 and end <= len(binary), "DDR overlap/alignment/bounds error")
        require(not any(binary[previous:start]), "DDR padding differs")
        expected = np.ascontiguousarray(exported["arrays"][row["key"]].astype(np.dtype(row["dtype"]), copy=False)).tobytes()
        require(binary[start:end] == expected and hashlib.sha256(expected).hexdigest() == row["sha256"], "DDR bytes differ")
        previous = end
    require(previous == len(binary), "Unindexed bytes after parameter payload")
    if loader["formal_eligible"]:
        require(loader["confirmation"] is not None, "Formal package needs confirmation evidence")
        report = replay_confirmation(read_json(loader["confirmation"], root), root=root)
        require(report["formal_eligible"] and report["candidate_identity_sha256"] == candidate["identity_sha256"],
                "Formal confirmation differs")
        exported["confirmation_report"] = report
    return loader, exported


def validate_board_return(candidate, package_path, board_receipt_path, *, root="."):
    """Actually compare raw full-graph board outputs; status strings are ignored.

    Board receipt: bindings, implementation_manifest, bitstream, loader_capture,
    raw_outputs (.npz), output_index (JSON), evidence_kind, device_id. Every file
    is a {path,sha256} reference. Implementation/capture must bind this loader and
    bitstream, all graph/quant/calibration/mapping/weight identities, and this ABI.
    """
    loader_path = artifact(package_path, root) if isinstance(package_path, dict) else Path(package_path).resolve()
    loader, exported = inspect_parameter_package(candidate, loader_path, root=root)
    path = artifact(board_receipt_path, root) if isinstance(board_receipt_path, dict) else Path(board_receipt_path).resolve()
    receipt = _json(path)
    require(receipt.get("schema_version") == BOARD_SCHEMA and receipt.get("test_evaluated") is False,
            "Invalid board receipt")
    expected = {**loader["bindings"], "loader_manifest_sha256": sha256_file(loader_path),
                "bitstream_sha256": receipt["bitstream"]["sha256"]}
    measurement_protocol = None
    if receipt.get("measurement_protocol") is not None:
        measurement_protocol = read_json(receipt["measurement_protocol"], root)
        require(measurement_protocol.get("schema_version") == MEASUREMENT_SCHEMA
                and measurement_protocol.get("frozen") is True, "Explicit frozen measurement protocol required")
        clock_hz = measurement_protocol.get("clock_hz")
        require(type(clock_hz) is int and clock_hz > 0 and measurement_protocol.get("target_device"),
                "Frozen measurement clock/device required")
        expected.update(measurement_protocol_sha256=receipt["measurement_protocol"]["sha256"],
                        clock_hz=clock_hz, target_device=measurement_protocol["target_device"])
    artifact(receipt["bitstream"], root)
    require(receipt["bindings"] == expected, "Board candidate/bitstream/loader/graph/quant/calibration/mapping/weight binding differs")
    implementation = read_json(receipt["implementation_manifest"], root)
    require(implementation.get("abi") == ABI and implementation.get("bindings") == expected,
            "Bitstream implementation does not bind this software ABI/graph")
    capture = read_json(receipt["loader_capture"], root)
    require(capture.get("bindings") == expected and capture.get("abi_observed") == ABI
            and capture.get("device_id") == receipt.get("device_id") and bool(receipt.get("device_id"))
            and capture.get("loaded_parameter_binary_sha256") == loader["binary"]["sha256"],
            "Raw loader capture does not prove selected bitstream/parameter load")
    raw_path = artifact(receipt["raw_outputs"], root)
    output_index = read_json(receipt["output_index"], root)
    require(output_index.get("bindings") == expected and output_index.get("input_images_sha256") ==
            exported["reference_index"]["input_images_sha256"], "Board input/reference identity differs")
    expected_rows = exported["reference_rows"]
    rows = {row["node"]: row for row in output_index["nodes"]}
    require(len(rows) == len(output_index["nodes"]) and set(rows) == set(expected_rows),
            "Board must return every graph node and input, not an operator-only smoke")
    comparisons = []
    with np.load(raw_path, allow_pickle=False) as actual, np.load(
            artifact(loader["reference_outputs"], root), allow_pickle=False) as reference_outputs:
        require(set(actual.files) == {row["key"] for row in rows.values()}, "Board raw output index differs")
        for name, row in rows.items():
            value, reference_value = actual[row["key"]], reference_outputs[expected_rows[name]["key"]]
            require(value.dtype == reference_value.dtype and value.shape == reference_value.shape
                    and list(value.shape) == row["shape"] and value.dtype == np.dtype(row["dtype"]),
                    "Board raw dtype/shape differs: " + name)
            require(_array_sha(value) == row["sha256"], "Board raw data differs from captured hash: " + name)
            comparisons.append({"node": name, "elements": int(value.size),
                                "mismatches": int(np.count_nonzero(value != reference_value)),
                                "actual_sha256": _array_sha(value), "reference_sha256": _array_sha(reference_value)})
    mismatches = sum(row["mismatches"] for row in comparisons)
    measured = receipt.get("evidence_kind") == implementation.get("evidence_kind") == capture.get("evidence_kind") == "board_measured"
    reviewed, review_reason = reviewed_board_capture(receipt, expected, root=root)
    formal_blockers = []
    if not loader["formal_eligible"] or candidate.get("purpose") != "formal":
        formal_blockers.append("CANDIDATE_AND_CONFIRMATION_NOT_FORMAL")
    if not measured:
        formal_blockers.append("EVIDENCE_IS_NOT_A_REAL_BOARD_CAPTURE")
    if measurement_protocol is None:
        formal_blockers.append("FROZEN_CLOCK_DEVICE_PROTOCOL_MISSING")
    if mismatches:
        formal_blockers.append("INTEGER_PARITY_FAILED")
    if not reviewed:
        formal_blockers.append(review_reason)
    formal = not formal_blockers
    software_reference = None
    if loader["formal_eligible"]:
        confirmation_request = exported["confirmation_report"]["request"]
        selected = [read_json(item, root) for item in confirmation_request["seed_receipts"]]
        selected = [item for item in selected if item["identity"]["identity_sha256"] == candidate["identity_sha256"]]
        require(len(selected) == 1, "Selected software accuracy reference missing from confirmation")
        software_reference = selected[0]["profiles"][candidate["profile"]]["records"]
    measurements = evaluate_measurements(receipt, expected, measurement_protocol, root=root, formal_base=formal,
                                         candidate=candidate, software_reference=software_reference)
    return {"schema_version": BOARD_SCHEMA, "status": "FULL_GRAPH_INTEGER_PARITY_PASS" if mismatches == 0 else "INTEGER_PARITY_FAILED",
            "formal_eligible": formal, "hardware_verified": formal, "evidence_kind": receipt.get("evidence_kind"),
            "scope": "full_graph_raw_integer_numerical_parity_only", "bindings": expected,
            "mismatches": mismatches, "nodes": comparisons, "board_receipt_sha256": sha256_file(path),
            "formal_gate": "PASS" if formal else "BLOCKED", "formal_blockers": formal_blockers,
            "measurements": measurements,
            "valid_accuracy_verified": measurements["accuracy"]["formal_eligible"],
            "latency_verified": measurements["latency"]["formal_eligible"],
            "power_verified": measurements["power"]["formal_eligible"],
            "deployment_ready": formal and all(row["formal_eligible"] for row in measurements.values()),
            "launch_hardware": False, "test_evaluated": False}


def reviewed_board_capture(receipt, expected_bindings, *, root="."):
    """A real board label is insufficient; require an admitted independent review."""
    if receipt.get("independent_review") is None:
        return False, "INDEPENDENT_BOARD_CAPTURE_REVIEW_MISSING"
    try:
        review_path = artifact(receipt["independent_review"], root)
        entry = {"bindings": expected_bindings,
                 "implementation_manifest_sha256": receipt["implementation_manifest"]["sha256"],
                 "loader_capture_sha256": receipt["loader_capture"]["sha256"],
                 "raw_outputs_sha256": receipt["raw_outputs"]["sha256"],
                 "output_index_sha256": receipt["output_index"]["sha256"],
                 "measurement_artifacts_sha256": digest(receipt.get("measurements", {})),
                 "independent_review_sha256": sha256_file(review_path)}
        if entry not in REGISTERED_FORMAL_BOARD_RETURNS:
            return False, "BOARD_CAPTURE_NOT_IN_INDEPENDENTLY_REVIEWED_REGISTRY"
        review = _json(review_path)
        bindings = {key: value for key, value in entry.items() if key != "independent_review_sha256"}
        if (review.get("schema_version") != "sonar192_independent_board_review_v1"
                or review.get("evidence_kind") != "independent_real_board_review"
                or review.get("bindings") != bindings or review.get("status") != "PASS_CAPTURE_REVIEW"
                or review.get("test_evaluated") is not False):
            return False, "REGISTERED_BOARD_REVIEW_CONTENT_DIFFERS"
        return True, None
    except (OSError, ValueError, KeyError, TypeError):
        return False, "INVALID_INDEPENDENT_BOARD_REVIEW_ARTIFACT"


def _finite(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _measurement_raw(receipt, stage, expected, root):
    data = read_json(receipt["measurements"][stage], root)
    require(data.get("bindings") == expected, stage + " candidate/bitstream/clock/protocol binding differs")
    require(data.get("evidence_kind") in ("board_measured", "synthetic_diagnostic"), stage + " evidence kind missing")
    return data


def _accuracy(data, protocol, root, candidate=None, software_reference=None):
    require(protocol.get("metric") == METRIC and protocol.get("split") == "valid"
            and protocol.get("selection_policy") == "epoch80_ema" and protocol.get("validation_count") == 99
            and protocol.get("num_classes") == 2, "Accuracy requires frozen full99 epoch80 EMA COCO protocol")
    require(data.get("scope") == "whole_network_host_decode_NMS" and data.get("test_evaluated") is False,
            "Accuracy scope must include whole network and host decode/NMS on valid only")
    require(_finite(protocol.get("maximum_absolute_drop")) and _finite(protocol.get("maximum_relative_drop")),
            "Explicit finite accuracy tolerances required")
    require(data.get("software_reference") == protocol.get("software_reference"), "Software accuracy reference differs")
    if candidate is not None and candidate["purpose"] == "formal":
        validate_validation_contract(protocol, artifact(protocol["execution_manifest"], root), candidate["checkpoint"])
        require(software_reference is not None and protocol["software_reference"]["sha256"] == software_reference["sha256"],
                "Accuracy software baseline is not the confirmed selected quantized checkpoint")
    software_rows = read_json(protocol["software_reference"], root)
    board_rows = read_json(data["records"], root)
    software, board = score_records(software_rows, protocol), score_records(board_rows, protocol)
    drop = software - board
    passed = drop <= protocol["maximum_absolute_drop"] and drop <= protocol["maximum_relative_drop"] * software
    return {"passed": passed, "software_mAP50_95": software, "board_mAP50_95": board,
            "absolute_drop": drop, "images": 99, "scope": data["scope"], "metric": METRIC}


def _latency(data, protocol):
    require(protocol.get("scope") == data.get("scope") == "end_to_end_input_to_host_detections"
            and protocol.get("unit") == data.get("unit") == "us", "Latency scope/unit mismatch")
    require(protocol.get("batch_size") == data.get("batch_size") == 1
            and data.get("concurrency") == 1 and data.get("timebase") == "monotonic_ns", "Latency batch/concurrency/timebase mismatch")
    count, warmup = protocol.get("repetitions"), protocol.get("warmup")
    require(type(count) is int and count >= 30 and type(warmup) is int and warmup >= 1
            and data.get("warmup_discarded") == warmup, "Latency needs frozen warmup and at least30 repeats")
    require(protocol.get("statistic") in ("mean", "p95") and _finite(protocol.get("limit_us"), positive=True),
            "Latency statistic and budget required")
    samples = data["samples"]
    require(len(samples) == count and [row["repeat"] for row in samples] == list(range(count)), "Latency repeated-sample count/order differs")
    durations, previous = [], -1
    for row in samples:
        begin, end = row["start_ns"], row["end_ns"]
        require(type(begin) is int and type(end) is int and begin >= previous and end > begin,
                "Latency timestamps overlap or reverse")
        duration = (end - begin) / 1000.
        require(_finite(row.get("duration_us"), positive=True) and abs(row["duration_us"] - duration) <= 1e-6,
                "Latency reported duration differs from raw monotonic timestamps")
        durations.append(duration)
        previous = end
    values = {"mean": float(np.mean(durations)), "p50": float(np.percentile(durations, 50)),
              "p95": float(np.percentile(durations, 95)), "maximum": max(durations)}
    return {"passed": values[protocol["statistic"]] <= protocol["limit_us"], "repetitions": count,
            "unit": "us", "scope": protocol["scope"], "statistics": values,
            "accepted_statistic": protocol["statistic"], "limit_us": protocol["limit_us"]}


def _integrate(samples, begin, end, minimum, maximum_gap):
    require(_finite(begin) and _finite(end, positive=True) and end > begin
            and type(minimum) is int and minimum >= 3 and _finite(maximum_gap, positive=True),
            "Power time window/sample protocol invalid")
    require(len(samples) >= minimum, "Power trace has too few samples")
    times, powers = [row["time_s"] for row in samples], [row["power_w"] for row in samples]
    require(all(_finite(value) for value in times + powers), "Power trace has nonfinite/negative samples")
    require(abs(times[0] - begin) <= 1e-9 and abs(times[-1] - end) <= 1e-9,
            "Power trace must cover the exact frozen time window")
    gaps = [b - a for a, b in zip(times, times[1:])]
    require(all(0 < gap <= maximum_gap for gap in gaps), "Power trace timestamps reverse or have missing intervals")
    return sum(gap * (a + b) / 2. for gap, a, b in zip(gaps, powers, powers[1:]))


def _power(data, protocol):
    require(protocol.get("unit") == data.get("unit") == "W" and protocol.get("time_unit") == data.get("time_unit") == "s",
            "Power units must be explicit W and s")
    require(protocol.get("scope") in ("board_input_total", "accelerator_rails")
            and data.get("scope") == protocol["scope"] and data.get("rails") == protocol.get("rails")
            and isinstance(protocol.get("rails"), list) and protocol["rails"]
            and len(set(protocol["rails"])) == len(protocol["rails"]), "Power rail/scope mismatch")
    require(protocol.get("measurement_device") == data.get("measurement_device") and bool(protocol["measurement_device"])
            and protocol.get("integration") == "trapezoid" and data.get("timestamp_origin") == "capture_start", "Power device/timebase/integration mismatch")
    require(protocol.get("idle_subtraction") in ("none", "subtract_measured_idle_time_weighted_mean"),
            "Power idle subtraction must be explicitly frozen")
    begin, end = protocol["window_start_s"], protocol["window_end_s"]
    minimum, gap = protocol["minimum_samples"], protocol["max_sample_gap_s"]
    energy = _integrate(data["samples"], begin, end, minimum, gap)
    count = protocol.get("completed_inferences")
    require(type(count) is int and count > 0, "Power inference denominator required")
    events = data["inference_events"]
    require(len(events) == count and [row["inference"] for row in events] == list(range(count)),
            "Power inference event count differs")
    previous = begin
    for row in events:
        require(_finite(row["start_s"]) and _finite(row["end_s"]) and previous <= row["start_s"]
                < row["end_s"] <= end, "Power inference event lies outside window or overlaps")
        previous = row["end_s"]
    idle_mean = 0.
    if protocol["idle_subtraction"] == "subtract_measured_idle_time_weighted_mean":
        idle_begin, idle_end = protocol["idle_window_start_s"], protocol["idle_window_end_s"]
        require(idle_end <= begin or idle_begin >= end, "Idle and active power windows overlap")
        idle_energy = _integrate(data["idle_samples"], idle_begin, idle_end, minimum, gap)
        idle_mean = idle_energy / (idle_end - idle_begin)
    else:
        require(data.get("idle_samples") in (None, []), "Undeclared idle correction data")
    duration = end - begin
    reported_energy = energy - idle_mean * duration
    require(reported_energy >= 0, "Idle-subtracted energy is negative; refuse clipping or inference")
    average, per_inference = reported_energy / duration, reported_energy / count
    require(_finite(protocol.get("max_average_power_w"), positive=True)
            and _finite(protocol.get("max_energy_per_inference_j"), positive=True), "Power/energy budgets required")
    return {"passed": average <= protocol["max_average_power_w"] and per_inference <= protocol["max_energy_per_inference_j"],
            "scope": protocol["scope"], "rails": protocol["rails"], "duration_s": duration,
            "idle_subtraction": protocol["idle_subtraction"], "idle_mean_w": idle_mean,
            "total_energy_j": energy, "reported_energy_j": reported_energy,
            "reported_average_power_w": average, "energy_per_inference_j": per_inference,
            "completed_inferences": count, "samples": len(data["samples"])}


def evaluate_measurements(receipt, expected_bindings, protocol, *, root=".", formal_base=False,
                          candidate=None, software_reference=None):
    """Recompute independent stages; absent or malformed evidence stays blocked.

    This can validate genuine future measurements. It does not estimate absent
    latency/power or infer whole-valid accuracy from a numerical smoke witness.
    """
    output = {}
    if formal_base:
        require(candidate is not None and validate_identity(candidate)["purpose"] == "formal",
                "Formal measurement admission requires the validated candidate")
    for stage in ("accuracy", "latency", "power"):
        if protocol is None or stage not in protocol or stage not in receipt.get("measurements", {}):
            output[stage] = {"status": "NOT_VERIFIED", "formal_eligible": False,
                             "reason": "Bound frozen protocol and raw " + stage + " evidence are required"}
            continue
        try:
            data = _measurement_raw(receipt, stage, expected_bindings, root)
            result = _accuracy(data, protocol[stage], root, candidate, software_reference) if stage == "accuracy" else (
                _latency(data, protocol[stage]) if stage == "latency" else _power(data, protocol[stage]))
            result["status"] = "ACCEPTANCE_PASS" if result.pop("passed") else "ACCEPTANCE_FAILED"
            result["formal_eligible"] = bool(formal_base and data["evidence_kind"] == "board_measured"
                                              and result["status"] == "ACCEPTANCE_PASS")
            result["raw_evidence_sha256"] = receipt["measurements"][stage]["sha256"]
            output[stage] = result
        except (ValueError, KeyError, TypeError, OSError) as error:
            output[stage] = {"status": "BLOCKED_INVALID_EVIDENCE", "formal_eligible": False,
                             "reason": str(error)}
    return output
