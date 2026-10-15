"""Read-only FPN history audit of nine already exported, bound PTQ packages.

This extracts existing scales; it neither observes images nor rebuilds a model.
Local mirrors are admitted by their original SHA, without rewriting bindings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.sonar_deploy import admission as gate


def summarize_fpn(graph, profile):
    """Compare exported Add replay against every archived observer record."""
    levels = list(graph["outputs"])
    count = len(levels) - 1
    records = profile["fpn_adds"]
    nodes = [n for n in graph["nodes"] if n["op"] == "upsample_add"]
    indices = [n["archived_fpn_record_index"] for n in nodes]
    history = graph["fpn_history"]
    if count not in (2, 3) or len(nodes) != count or len(records) % count:
        raise ValueError("FPN physical Add/observer record count mismatch")
    if (indices != list(reversed(range(count))) or
            history["policy"] != "replay_archived_wrapper_index_without_recalibration" or
            history["saved_record_count"] != len(records) or
            history["used_record_indices"] != list(range(count)) or
            history["unconsumed_records"] != len(records) - count):
        raise ValueError("Exported FPN history differs from archived wrapper indexing")
    mapping = []
    for node, index in zip(nodes, indices):
        spec = records[index]
        if node["id"] != f"neck.fuse.{index}" or any(
                node[graph_key] != spec[profile_key] for graph_key, profile_key in
                (("input_exp_a", "left_exp"), ("input_exp_b", "right_exp"), ("output_exp", "output_exp"))):
            raise ValueError("Exported Add does not consume the saved exponent record")
        observed_level = levels[count - 1 - index]
        mapping.append(dict(node=node["id"], consumer_level=levels[index], record_index=index,
                            observer_batch_index=0, observer_level_from_append_order=observed_level,
                            level_order_reversed=observed_level != levels[index], **spec))
    groups = defaultdict(list)
    for i, spec in enumerate(records):
        groups[tuple(spec[k] for k in ("left_exp", "right_exp", "output_exp"))].append(i)
    repeated = [dict(exponents=list(k), record_indices=v) for k, v in sorted(groups.items()) if len(v) > 1]
    return dict(saved_record_count=len(records), physical_add_count=count,
                observer_batch_count=len(records) // count,
                observer_append_level_order=list(reversed(levels[:-1])),
                wrapper_execution_record_indices=indices,
                used_record_indices=sorted(indices), unused_record_indices=list(range(count, len(records))),
                unused_record_count=len(records) - count,
                unique_exponent_triplet_count=len(groups),
                repeated_exponent_record_count=len(records) - len(groups),
                repeated_exponent_groups=repeated, wrapper_mapping=mapping,
                mismapped_level_count=sum(m["level_order_reversed"] for m in mapping),
                recalibrated=False, historical_order_corrected=False, test_evaluated=False)


def audit_profiles(root, profiles_root, calibration_manifest, output):
    root, profiles_root, output = map(lambda p: Path(p).resolve(), (root, profiles_root, output))
    binding = gate.read_json(root / "DEPLOYMENT_BINDING.json")
    if binding.get("status") != "FROZEN" or binding.get("schema_version") != gate.SCHEMA:
        raise ValueError("Frozen deployment binding required")
    gate._sealed(binding)
    runs = binding["runs"]
    if len(runs) != 9 or {(r["network"], r["seed"]) for r in runs} != {(n, s) for n in gate.NETWORKS for s in gate.SEEDS}:
        raise ValueError("Exactly nine original B1/C/D seed42/43/44 bindings required")
    calibration_path = gate._file(ROOT, calibration_manifest,
        binding["source_files"]["calibration_manifest"]["sha256"], "original calibration manifest")
    calibration = gate.read_json(calibration_path)
    if (calibration.get("split") != "train" or calibration.get("test_evaluated") is not False or
            calibration.get("augmentation") is not False or calibration.get("shuffle") is not False):
        raise ValueError("Frozen train-only deterministic calibration required")
    rows = []
    for identity in sorted(runs, key=lambda r: (gate.NETWORKS.index(r["network"]), r["seed"])):
        network, seed = identity["network"], identity["seed"]
        if identity["architecture_id"] != gate.ARCHITECTURES[network]:
            raise ValueError("Architecture identity differs")
        folder = root / network / f"seed_{seed}"
        result = gate.read_json(folder / "result.json")
        package = folder / "package"
        wrapper_path = gate._file(package, "manifest.json", result["package_manifest_sha256"], "package wrapper")
        wrapper = gate.read_json(wrapper_path)
        for key in gate.IDENTITY:
            if result[key] != identity[key] or wrapper[key] != identity[key]:
                raise ValueError("Result/package/deployment identity mismatch: " + key)
        for filename, digest in wrapper["files"].items():
            gate._file(package, filename, digest, "exported package artifact")
        graph = gate.read_json(package / "graph.json")
        if any(graph["provenance"][key] != identity[key] for key in gate.IDENTITY):
            raise ValueError("Graph provenance mismatch")
        if graph.get("test_evaluated") is not False or not graph.get("real_weights_bound"):
            raise ValueError("Only real, test-sealed exported graphs allowed")
        profile_path = gate._file(profiles_root, f"{network}/seed_{seed}/h2_ptq_profile.json",
                                  identity["profile_sha256"], "archived profile mirror")
        profile = gate.read_json(profile_path)
        summary = summarize_fpn(graph, profile)
        if summary["observer_batch_count"] * calibration["batch_size"] != calibration["number_of_calibration_images"]:
            raise ValueError("Observer record cycles differ from frozen calibration batch count")
        rows.append(dict(network=network, seed=seed, architecture_id=identity["architecture_id"],
            checkpoint_sha256=identity["checkpoint_sha256"], profile_sha256=identity["profile_sha256"],
            graph_sha256=gate.sha256(package / "graph.json"),
            calibration_sha256=gate.sha256(calibration_path), **summary))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "FPN_HISTORY_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":"), ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    lines = ["# FPN historical calibration-record audit", "",
        "Scope: nine existing real PTQ packages; no calibration, model inference, scale correction, or test access.", "",
        "Graph, package binaries and mirrored archived profiles were checked against the original deployment SHA bindings. "
        "This read-only extraction does not rerun full checkpoint admission; it does not rewrite absolute server paths in the frozen manifest.", "",
        "| Network | Seed | Saved records | Physical Adds | Observer batches | Consumed indices (execution order) | Unused | Unique exponent triples | Repeated exponent records |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['network']} | {row['seed']} | {row['saved_record_count']} | {row['physical_add_count']} | "
                     f"{row['observer_batch_count']} | {row['wrapper_execution_record_indices']} | {row['unused_record_count']} | "
                     f"{row['unique_exponent_triplet_count']} | {row['repeated_exponent_record_count']} |")
    lines += ["", "The historical observer appended each batch's Add observations in descending feature-level order. "
        "It kept all 16 batch cycles rather than aggregating each physical Add across batches. "
        "The historical wrapper executes descending level index but reads `add_exponents[index]`; only the first physical-Add-count entries are used.", "",
        "Consequently, observer order and consumer level do not coincide for the outer levels. "
        "The deployed graph intentionally replays the historical wrapper indices exactly. "
        "An observer level below is reconstructed from the inspected loop order, not a level tag stored in the profile.", "",
        "| Network | Consumer Add level | Saved index | Observer level from append order |",
        "| --- | --- | --- | --- |"]
    for network in gate.NETWORKS:
        for mapping in next(r for r in rows if r["network"] == network)["wrapper_mapping"]:
            lines.append(f"| {network} | {mapping['consumer_level']} | {mapping['record_index']} | {mapping['observer_level_from_append_order']} |")
    lines += ["", "Repeated exponent triples count equal numeric values, not duplicated images or identical underlying activations. "
        "The CSV retains the exact repeated index groups, every unused index, and per-run consumed exponents. "
        "Unused entries are preserved in the original profiles; they are not deleted or silently substituted.", "",
        "No conclusion is made here about how a corrected calibration would affect AP. Such a correction would define a new protocol.", "",
        f"Deployment binding SHA256: `{gate.sha256(root / 'DEPLOYMENT_BINDING.json')}`.",
        f"Inspected wrapper/observer source: `src/hwnas_fpga/deploy/h2_ptq.py`, SHA256 `{gate.sha256(ROOT / 'src/hwnas_fpga/deploy/h2_ptq.py')}`.",
        f"Audit source SHA256: `{gate.sha256(__file__)}`.",
        "`recalibrated=false`; `historical_order_corrected=false`; `test_evaluated=false`.", ""]
    (output / "FPN_HISTORY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(status="READ_ONLY_AUDIT_COMPLETE", runs=len(rows), output=str(output), test_evaluated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profiles-root", type=Path, default=ROOT / "artifacts/phase5a_quantization")
    parser.add_argument("--calibration-manifest", type=Path, default=ROOT / "reports/phase5a_quantization/CALIBRATION_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/sonar_real_int8_deployment_v1")
    args = parser.parse_args()
    print(json.dumps(audit_profiles(args.root, args.profiles_root, args.calibration_manifest, args.output), indent=2))


if __name__ == "__main__":
    main()
