"""Fixed train-prefix traces for real PTQ software/C/RTL/board parity.

This module never opens valid/test. Full valid evaluation lives in run.py.
The independent NumPy oracle can run separately because full detector INT64
convolution is intentionally slow and must not be silently replaced by FP32.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tools/sonar_analysis"), str(ROOT / "scripts")]

from hwnas_fpga.deploy.h2_ptq import H2FakeQuantConv, aligned_add_dequant
from tools.sonar_deploy.ptq_export import IntegerGraphModel, _load_package, restore_archived_ptq
from tools.fpga_h2 import numeric


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write(path, payload):
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _train_prefix(manifest):
    """Read exactly the first eight images using the archived no-aug loader."""
    from phase5a_quantization.common import baseline
    import baseline as frozen
    if (manifest.get("split") != "train" or manifest.get("test_evaluated") is not False or
            manifest.get("augmentation") is not False or manifest.get("shuffle") is not False or
            manifest.get("calibration_seed") != 42 or manifest.get("batch_size") != 4 or
            len(manifest["entries"]) != 64):
        raise ValueError("requires frozen train-only64 calibration manifest")
    data = frozen.resolve(baseline()["dataset"]["data_dir"])
    # Keep lexical split membership: the isolated deployment root intentionally
    # mounts only train/valid symlinks. Resolving each image before relative_to
    # would jump to the source root and falsely reject this sealed layout.
    loader = frozen.make_loader(baseline(), data, "train", 42, 0, train=False)
    dataset = loader.dataset
    actual = [Path(path).relative_to(dataset.root).as_posix() for path in dataset.samples[:64]]
    expected = [entry["image"].replace("\\", "/") for entry in manifest["entries"]]
    if actual != expected:
        raise ValueError("calibration image order drifted")
    for entry in manifest["entries"]:
        for key in ("image", "annotation"):
            path = Path(entry[key])
            if path.is_absolute() or path.parts[0] != "train" or ".." in path.parts:
                raise ValueError("non-train witness entry")
            if sha(data / path) != entry[key + "_sha256"]:
                raise ValueError("frozen train witness SHA mismatch")
    for index in range(8):
        image, _target = dataset[index]
        yield index, manifest["entries"][index], image.unsqueeze(0)


@torch.no_grad()
def _historical_trace(model, image, graph):
    """Instrument a private restored copy; preserve old FPN forward indexing."""
    records, handles = {}, []
    bridges = {n["id"]: n for n in graph["nodes"] if n["op"] == "requant"}
    for name, module in model.named_modules():
        if not isinstance(module, H2FakeQuantConv):
            continue
        def capture(_m, _inputs, out, name=name):
            records[name] = out.detach().cpu().clone()
        handles.append(module.register_forward_hook(capture))
        bridge_name = name + ".input_requant"
        if bridge_name in bridges:
            exp = bridges[bridge_name]["output_exp"]
            def capture_input(_m, inputs, name=bridge_name, exp=exp):
                q = torch.round(inputs[0].to(torch.float32) / (2.0 ** exp)).clamp(-128, 127)
                records[name] = (q * (2.0 ** exp)).detach().cpu()
            handles.append(module.register_forward_pre_hook(capture_input))
    neck, original = model.neck, model.neck.forward
    def traced_neck(features):
        fused = [conv(feature) for conv, feature in zip(neck.lateral_convs, features)]
        for index in range(len(fused) - 2, -1, -1):
            upsampled = F.interpolate(fused[index + 1], size=fused[index].shape[-2:], mode="nearest")
            spec = neck.add_exponents[index]
            for side, value in (("left", fused[index]), ("right", fused[index + 1])):
                key = f"neck.fuse.{index}.{side}_requant"
                if key in bridges:
                    exp = spec[side + "_exp"]
                    q = torch.round(value.to(torch.float32) / (2.0 ** exp)).clamp(-128, 127)
                    records[key] = (q * (2.0 ** exp)).detach().cpu()
            fused[index] = aligned_add_dequant(fused[index], upsampled, spec["left_exp"], spec["right_exp"], spec["output_exp"])
            records[f"neck.fuse.{index}"] = fused[index].detach().cpu().clone()
        return tuple(conv(value) for conv, value in zip(neck.output_convs, fused))
    neck.forward = traced_neck
    try:
        model(image)
    finally:
        neck.forward = original
        for handle in handles:
            handle.remove()
    return records


def prepare_train_witnesses(package, model, profile, calibration_manifest, output, *, device="cuda"):
    package, calibration_manifest, output = Path(package), Path(calibration_manifest), Path(output)
    calibration = json.loads(calibration_manifest.read_text(encoding="utf-8"))
    integer = IntegerGraphModel(package, model).to(device).eval()
    graph = integer.graph
    if sha(calibration_manifest) != graph["provenance"]["calibration_sha256"]:
        raise ValueError("calibration manifest not bound to package")
    # Phase5A folded BN after moving to CUDA. Folding on CPU can move a value
    # across an integer rounding tie (observed D44 bias / B1 seed43 weight).
    # Reproduce the archived backend; this does not change scales/calibration.
    _, historical = restore_archived_ptq(model.to(device), profile)
    historical.to(device).eval()
    output.mkdir(parents=True, exist_ok=False)
    frames, differences = [], []
    started = time.perf_counter()
    for index, entry, image in _train_prefix(calibration):
        image = image.to(device)
        _heads, values = integer.forward_quantized(image, retain=True)
        folder = output / f"frame_{index:03d}"
        folder.mkdir()
        traces = folder / "traces"; traces.mkdir()
        inp = values["image"][0].cpu().numpy()
        np.save(folder / "input.npy", inp, allow_pickle=False)
        (folder / "input_hwc.bin").write_bytes(inp.transpose(1, 2, 0).copy().tobytes())
        expected, layers = bytearray(), []
        reference = _historical_trace(historical, image, graph) if index == 0 else None
        for sequence, node in enumerate(graph["nodes"]):
            value = values[node["id"]][0].cpu().numpy()
            stem = f"{sequence:03d}_{node['id']}"
            np.save(traces / (stem + ".npy"), value, allow_pickle=False)
            hwc = value.transpose(1, 2, 0).copy().tobytes()
            layers.append(dict(index=sequence, node=node["id"], offset=len(expected), bytes=len(hwc),
                               sha256=hashlib.sha256(hwc).hexdigest(), chw_path=f"traces/{stem}.npy"))
            expected.extend(hwc)
            if reference is not None:
                if node["id"] not in reference:
                    raise ValueError("historical trace missing node: " + node["id"])
                floating = reference[node["id"]][0].numpy()
                dequantized = value.astype(np.float64) * (2.0 ** graph["edges"][node["id"]]["exp"])
                diff = np.abs(floating.astype(np.float64) - dequantized)
                differences.append(dict(index=sequence, node=node["id"], op=node["op"],
                                        num_values=diff.size, mismatch_count=int(np.count_nonzero(diff)),
                                        max_abs_error=float(diff.max()), mean_abs_error=float(diff.mean())))
        (folder / "expected_layers.bin").write_bytes(expected)
        write(folder / "layers.json", layers)
        for level, node in graph["outputs"].items():
            value = values[node][0].cpu().numpy()
            (folder / f"{level}_hwc.bin").write_bytes(value.transpose(1, 2, 0).copy().tobytes())
        files = {p.relative_to(output).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()}
        frames.append(dict(frame=index, source=entry, folder=folder.name, files=files,
                           input_path=f"{folder.name}/input.npy", input_hwc_path=f"{folder.name}/input_hwc.bin",
                           expected_layers_path=f"{folder.name}/expected_layers.bin"))
        write(output / "progress.json", dict(stage="TRAIN_PREFIX_TRACE", completed=index+1, total=8, test_evaluated=False))
    with (output / "historical_vs_integer_layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
        writer.writeheader(); writer.writerows(differences)
    first = next((row for row in differences if row["mismatch_count"]), None)
    write(output / "HISTORICAL_VS_INTEGER.json", dict(
        scope="diagnostic_only_independent_integer_valid_required", first_difference=first,
        total_node_mismatches=sum(r["mismatch_count"] for r in differences), layers=differences,
        graph_sha256=sha(package / "graph.json"), calibration_manifest_sha256=sha(calibration_manifest),
        sample=calibration["entries"][0], test_evaluated=False))
    manifest = dict(version=1, split="train", count=8, graph_sha256=sha(package / "graph.json"),
                    package_manifest_sha256=sha(package / "export_manifest.json"),
                    calibration_manifest_sha256=sha(calibration_manifest), frames=frames,
                    source_sha256=sha(__file__), numeric_source_sha256=sha(ROOT / "tools/fpga_h2/numeric.py"),
                    test_evaluated=False, elapsed_seconds=time.perf_counter()-started,
                    reference="INT8/INT32 canonical using bounded exact FP64 im2col dots; INT64 requant")
    manifest["files"] = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    write(output / "TRACE_MANIFEST.json", manifest)
    return manifest


def verify_numpy_first_image(package, trace_root, output):
    """Execute every full-size node independently in canonical NumPy INT64."""
    package, trace_root, output = Path(package), Path(trace_root), Path(output)
    manifest = json.loads((trace_root / "TRACE_MANIFEST.json").read_text())
    if manifest.get("split") != "train" or manifest.get("count") != 8 or manifest.get("test_evaluated") is not False:
        raise ValueError("NumPy parity requires fixed train witness set")
    if sha(package / "graph.json") != manifest["graph_sha256"]:
        raise ValueError("trace graph binding mismatch")
    frame = manifest["frames"][0]
    for name, digest in frame["files"].items():
        if sha(trace_root / name) != digest:
            raise ValueError("trace file hash mismatch: " + name)
    graph, params = _load_package(package)
    values = {"image": np.load(trace_root / frame["input_path"], allow_pickle=False)}
    layers = json.loads((trace_root / frame["folder"] / "layers.json").read_text())
    rows = []; started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    for index, node in enumerate(graph["nodes"]):
        begin = time.perf_counter(); a = values[node["inputs"][0]]
        if node["op"] == "conv":
            w, b = params[node["id"]]; value = numeric.conv(a, w, b, node)
        elif node["op"] == "requant":
            value = numeric.requantize_int32_to_int8(a, node["multiplier"], node["shift"])
        elif node["op"] in ("add", "upsample_add"):
            b = values[node["inputs"][1]]
            if node["op"] == "upsample_add": b = np.repeat(np.repeat(b, 2, axis=1), 2, axis=2)
            value = numeric.aligned_add(a, b, node["input_exp_a"], node["input_exp_b"], node["output_exp"])
        else:
            raise ValueError("unsupported NumPy oracle op")
        values[node["id"]] = value
        ref = np.load(trace_root / frame["folder"] / layers[index]["chw_path"], allow_pickle=False)
        error = np.abs(value.astype(np.int16) - ref.astype(np.int16))
        rows.append(dict(index=index, node=node["id"], mismatch_count=int(np.count_nonzero(error)),
                         max_abs_error=int(error.max()), elapsed_seconds=time.perf_counter()-begin))
        write(output / "progress.json", dict(completed=index+1, total=len(graph["nodes"]),
                                               mismatch_count=sum(r["mismatch_count"] for r in rows), test_evaluated=False))
    result = dict(status="PASS" if not any(r["mismatch_count"] for r in rows) else "FAIL",
                  scope="full_first_train_image_every_graph_node", rows=rows,
                  mismatch_count=sum(r["mismatch_count"] for r in rows), graph_sha256=sha(package / "graph.json"),
                  trace_manifest_sha256=sha(trace_root / "TRACE_MANIFEST.json"), elapsed_seconds=time.perf_counter()-started,
                  numeric_source_sha256=sha(ROOT / "tools/fpga_h2/numeric.py"), source_sha256=sha(__file__), test_evaluated=False)
    write(output / "NUMPY_PARITY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("trace", "numpy-check"))
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "numpy-check":
        if args.trace_root is None: parser.error("numpy-check requires --trace-root")
        result = verify_numpy_first_image(args.package, args.trace_root, args.output)
    else:
        if any(v is None for v in (args.checkpoint, args.profile, args.calibration)):
            parser.error("trace requires --checkpoint --profile --calibration")
        from phase5a_quantization.common import build_network
        from phase5a_quantization.run import load_weights
        graph, _ = _load_package(args.package)
        provenance = graph["provenance"]
        if sha(args.checkpoint) != provenance["checkpoint_sha256"] or sha(args.profile) != provenance["profile_sha256"]:
            raise ValueError("checkpoint/profile provenance mismatch")
        model, aid = build_network(provenance["network"])
        if aid != provenance["architecture_id"]: raise ValueError("architecture provenance mismatch")
        load_weights(model, args.checkpoint)
        result = prepare_train_witnesses(args.package, model, args.profile, args.calibration, args.output, device=args.device)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "files", "rows")}, indent=2))


if __name__ == "__main__":
    main()
