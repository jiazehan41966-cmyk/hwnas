"""One frozen train image: archived float PTQ versus exported integer graph.

Diagnostic only. Never recalibrates, evaluates valid/test, runs the NumPy
oracle, writes large traces, or changes an already bound export implementation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"),
               str(ROOT / "tools/sonar_analysis"), str(ROOT / "scripts")]
from tools.sonar_deploy import admission as gate
from tools.sonar_deploy import parity
from tools.sonar_deploy.ptq_export import IntegerGraphModel, restore_archived_ptq

SCHEMA = "real-int8-historical-layer-diagnostic-v1"
FIELDS = ("index", "node", "op", "output_exp", "num_values", "mismatch_count",
          "max_abs_error", "mean_abs_error", "integer_lsb")


def _numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def compare_layers(graph, historical, integer):
    """Exact element comparison in dequantized units, preserving graph order."""
    rows = []
    for index, node in enumerate(graph["nodes"]):
        name = node["id"]
        if name not in historical or name not in integer:
            raise ValueError("diagnostic trace missing node: " + name)
        floating, quantized = _numpy(historical[name]), _numpy(integer[name])
        shape = (1, *graph["edges"][name]["shape"])
        if floating.shape != shape or quantized.shape != shape:
            raise ValueError("diagnostic shape mismatch: " + name)
        if not np.issubdtype(floating.dtype, np.floating) or quantized.dtype != np.int8:
            raise ValueError("diagnostic expects float historical and INT8 graph values: " + name)
        if not np.isfinite(floating).all():
            raise ValueError("nonfinite historical trace: " + name)
        exponent = graph["edges"][name]["exp"]
        step = 2.0 ** exponent
        if not np.isfinite(step) or step <= 0:
            raise ValueError("invalid dequantization scale: " + name)
        difference = np.abs(floating.astype(np.float64) - quantized.astype(np.float64) * step)
        rows.append(dict(index=index, node=name, op=node["op"], output_exp=exponent,
                         num_values=int(difference.size), mismatch_count=int(np.count_nonzero(difference)),
                         max_abs_error=float(difference.max()), mean_abs_error=float(difference.mean()),
                         integer_lsb=step))
    if not rows:
        raise ValueError("empty diagnostic graph")
    return rows


def summarize(rows):
    return dict(first_difference=next((row for row in rows if row["mismatch_count"]), None),
                mismatch_count=sum(row["mismatch_count"] for row in rows),
                compared_values=sum(row["num_values"] for row in rows),
                max_abs_error=max(row["max_abs_error"] for row in rows),
                mean_abs_error=sum(row["mean_abs_error"] * row["num_values"] for row in rows) /
                               sum(row["num_values"] for row in rows),
                mismatch_definition="exact float64 difference after power-of-two dequantization; no tolerance",
                count_scope="sum over graph-node outputs, including propagated differences; not unique causes")


def _csv(path, rows, fields):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def _sources():
    paths = [Path(__file__), Path(parity.__file__), ROOT / "tools/sonar_deploy/ptq_export.py",
             ROOT / "tools/fpga_h2/numeric.py"]
    return {str(path.relative_to(ROOT)): gate.sha256(path) for path in paths}


def _diagnose_one(root, output, row, result, entry, image, device, binding_hash, sources):
    from phase5a_quantization.common import build_network
    from phase5a_quantization.run import load_weights
    import baseline as frozen
    package_path = gate._path(root, result["package_manifest_path"])
    package = package_path.parent
    graph_path = package / "graph.json"
    identity = {key: result[key] for key in gate.IDENTITY}
    expected = dict(**identity, deployment_binding_sha256=binding_hash,
                    package_manifest_sha256=result["package_manifest_sha256"],
                    graph_sha256=gate.sha256(graph_path), sample=entry, source_sha256=sources)
    folder = output / row["network"] / f"seed_{row['seed']}"
    report_path, csv_path = folder / "HISTORICAL_VS_INTEGER.json", folder / "historical_vs_integer_layers.csv"
    if report_path.exists():
        previous = gate.read_json(report_path)
        if (previous.get("status") != "COMPLETE" or previous.get("schema_version") != SCHEMA or
                any(previous.get(key) != value for key, value in expected.items()) or
                previous.get("layers_csv_sha256") != gate.sha256(csv_path)):
            raise ValueError("existing diagnostic identity/source/CSV changed; use a new output directory")
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        raise ValueError("partial prior diagnostic retained; use a new output directory")
    frozen.seed_all(row["seed"])
    begin = time.perf_counter()
    model, architecture = build_network(row["network"])
    if architecture != row["architecture_id"]:
        raise ValueError("diagnostic architecture identity changed")
    load_weights(model, gate._path(root, row["checkpoint_path"]))
    # Match Phase5A's backend: move to CUDA BEFORE BN folding. Saved profile
    # exponents and the historical FPN index order are replayed, not corrected.
    model.to(device).eval()
    folded, historical = restore_archived_ptq(model, gate._path(root, row["profile_path"]))
    historical.to(device).eval()
    integer = IntegerGraphModel(package, model).to(device).eval()
    graph = integer.graph
    if any(graph["provenance"].get(key) != expected[key] for key in gate.IDENTITY):
        raise ValueError("integer graph provenance differs from frozen deployment")
    if graph["provenance"].get("parameter_preparation_device") != "cuda":
        raise ValueError("archived diagnostic requires CUDA-folded package")
    with torch.inference_mode():
        tensor = image.to(device)
        historical_values = parity._historical_trace(historical, tensor, graph)
        _heads, integer_values = integer.forward_quantized(tensor, retain=True)
        rows = compare_layers(graph, historical_values, integer_values)
    report = dict(schema_version=SCHEMA, status="COMPLETE", **expected, **summarize(rows),
                  scope="diagnostic_only_independent_integer_valid_required", split="train", sample_count=1,
                  sample_index=0, calibration_recomputed=False, numpy_oracle_run=False, traces_saved=False,
                  layers=rows, historical_backend="CUDA folding and CUDA float fake-quant forward",
                  integer_backend="existing exported graph; bounded exact FP64 dots and integer requant",
                  torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(torch.device(device)),
                  elapsed_seconds=time.perf_counter() - begin,
                  created_utc=datetime.now(timezone.utc).isoformat(), test_evaluated=False)
    _csv(csv_path, rows, FIELDS)
    report["layers_csv_sha256"] = gate.sha256(csv_path)
    gate.atomic_json(report_path, report)
    del model, folded, historical, integer, tensor, historical_values, integer_values, _heads
    torch.cuda.empty_cache()
    return report


def run(root, output, *, device="cuda", networks=gate.NETWORKS, seeds=gate.SEEDS):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA_ENVIRONMENT_NOT_READY: original historical BN folding must use CUDA")
    networks, seeds = tuple(networks), tuple(seeds)
    if (not networks or not seeds or len(set(networks)) != len(networks) or len(set(seeds)) != len(seeds) or
            not set(networks).issubset(gate.NETWORKS) or not set(seeds).issubset(gate.SEEDS)):
        raise ValueError("only unique B1/C/D and seeds42/43/44 diagnostics are admitted")
    torch.set_num_threads(2)
    binding = gate.load_binding(root)
    binding_hash = gate.sha256(root / "DEPLOYMENT_BINDING.json")
    calibration_path = gate._path(root, binding["source_files"]["calibration_manifest"]["path"])
    calibration = gate.read_json(calibration_path)
    # Reuse the original loader/order/SHA checks, but consume only its first image.
    prefix = parity._train_prefix(calibration)
    index, entry, image = next(prefix); prefix.close()
    if index != 0 or entry != calibration["entries"][0]:
        raise ValueError("diagnostic must use the first frozen train calibration image")
    sources = _sources(); reports = []
    for network in networks:
        for seed in seeds:
            row = next(r for r in binding["runs"] if r["network"] == network and r["seed"] == seed)
            if row["calibration_sha256"] != gate.sha256(calibration_path):
                raise ValueError("diagnostic calibration manifest SHA differs")
            result = gate.load_result(root, row)
            report = _diagnose_one(root, output, row, result, entry, image, device, binding_hash, sources)
            reports.append(report)
            print(f"DIAGNOSTIC_COMPLETE {network} seed{seed} first={report['first_difference']['node'] if report['first_difference'] else 'NONE'}", flush=True)
    summary_rows = []
    for report in reports:
        first = report["first_difference"]
        summary_rows.append(dict(network=report["network"], seed=report["seed"],
            first_difference=first["node"] if first else "NONE", first_mismatch_count=first["mismatch_count"] if first else 0,
            first_max_abs_error=first["max_abs_error"] if first else 0.,
            mismatch_count=report["mismatch_count"], max_abs_error=report["max_abs_error"],
            mean_abs_error=report["mean_abs_error"], graph_sha256=report["graph_sha256"],
            checkpoint_sha256=report["checkpoint_sha256"], test_evaluated=False))
    _csv(output / "HISTORICAL_VS_INTEGER_SUMMARY.csv", summary_rows, tuple(summary_rows[0]))
    manifest = dict(schema_version=SCHEMA, status="COMPLETE", completed=len(reports),
        expected=len(networks)*len(seeds), all_nine_complete=len(reports)==9, deployment_root=str(root),
        deployment_binding_sha256=binding_hash, source_sha256=sources, sample=entry, sample_count=1,
        scope="float_simulation_vs_integer_diagnostic_not_FPGA_parity", test_evaluated=False,
        reports={f"{r['network']}/seed_{r['seed']}/HISTORICAL_VS_INTEGER.json":
                 gate.sha256(output / r["network"] / f"seed_{r['seed']}" / "HISTORICAL_VS_INTEGER.json") for r in reports},
        summary_sha256=gate.sha256(output / "HISTORICAL_VS_INTEGER_SUMMARY.csv"))
    gate.atomic_json(output / "DIAGNOSTIC_MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Active frozen deployment root")
    parser.add_argument("--output", type=Path, help="Default: reports/sonar_real_int8_deployment_v1/diagnostic_layers/<root-name>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--network", choices=gate.NETWORKS, action="append")
    parser.add_argument("--seed", choices=gate.SEEDS, type=int, action="append")
    args = parser.parse_args()
    output = args.output or ROOT / "reports/sonar_real_int8_deployment_v1/diagnostic_layers" / args.root.resolve().name
    report = run(args.root, output, device=args.device, networks=args.network or gate.NETWORKS, seeds=args.seed or gate.SEEDS)
    print(f"{report['status']}: {report['completed']}/{report['expected']}; test_evaluated=false; {output}")


if __name__ == "__main__":
    main()
