#!/usr/bin/env python3
"""Regenerate protocol_summary.json from existing per-record JSON files.

Used when a protocol run was assembled from separately-trained records (e.g.
resumed/patched after an interruption) so the aggregate summary and
claimability need to be rebuilt without retraining. Pure aggregation over
the run_fold*_seed*.json records already present in the run directory.

Note on provenance: records assembled from patch runs may carry mixed
run_fingerprints. This finalizer recomputes the aggregate but never fabricates
a single fingerprint; mixed or missing fingerprints remain non-claimable.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.data.dataset import NKSID_CLASSES
from hwnas_fpga.training.protocol_reporting import canonical_sha256, protocol_claimability

RECORD_RE = re.compile(r"run_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json$")


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0, "values": []}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
        "values": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-provenance", default=None,
                        help="override; else read from run_manifest.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    record_paths = sorted(glob.glob(str(run_dir / "run_fold*_seed*.json")))
    if not record_paths:
        print(f"No records in {run_dir}", file=sys.stderr)
        return 1

    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in record_paths]
    folds = sorted({r["fold"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})

    selection_provenance = args.selection_provenance
    manifest_path = run_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if selection_provenance is None and manifest_path.exists():
        selection_provenance = manifest.get("immutable_config", {}).get(
            "selection_provenance"
        ) or manifest.get("claimability", {}).get("selection_provenance")
    selection_provenance = selection_provenance or "baseline_predeclared"
    protocol_context_sha256 = manifest.get("protocol_context_sha256")
    if not protocol_context_sha256:
        immutable_config = manifest.get("immutable_config") or {}
        if immutable_config:
            protocol_context_sha256 = canonical_sha256(
                {key: value for key, value in immutable_config.items() if key != "candidate"}
            )

    provenance_complete = all(
        len(str((r.get("checkpoint") or {}).get("sha256", ""))) == 64
        and len(str(r.get("split_sha256", ""))) == 64
        for r in runs
    )
    claimability = protocol_claimability(
        folds=folds,
        seeds=seeds,
        completed_pairs=[(r["fold"], r["seed"]) for r in runs],
        selection_provenance=selection_provenance,
        outer_validation_used_for_selection=False,
        provenance_complete=provenance_complete,
        provenance_fingerprints=[str(r.get("run_fingerprint", "")) for r in runs],
        group_split_available=bool(
            (manifest.get("data_protocol") or {}).get("group_split_available", False)
        ),
        protocol_context_sha256=protocol_context_sha256,
        provenance_contexts=[str(r.get("protocol_context_sha256", "")) for r in runs],
    )

    class_names = list(NKSID_CLASSES)
    have_pcf = all("outer_per_class_f1" in r for r in runs)
    aggregate = {
        "protocol": "nksid_outer5fold_inner_contiguous_v1",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_dir.name,
        "finalized_from_records": True,
        "claimability": claimability,
        "recipe": runs[0].get("recipe")
        or (manifest.get("immutable_config", {}).get("recipe") if manifest else None),
        "protocol_context_sha256": protocol_context_sha256,
        "normalization": (manifest.get("data_protocol") or {}).get(
            "normalization",
            (manifest.get("immutable_config") or {}).get("normalization"),
        ),
        "group_split_available": bool(
            (manifest.get("data_protocol") or {}).get("group_split_available", False)
        ),
        "group_generalization_claimable": bool(
            (manifest.get("data_protocol") or {}).get(
                "group_generalization_claimable", False
            )
        ),
        "legacy_result": selection_provenance == "legacy_fold0_selected",
        "claim_boundary": (
            "Legacy selected benchmark only; it cannot establish NAS method "
            "generalization or group-safe cross-task generalization."
            if selection_provenance == "legacy_fold0_selected"
            else "Formal fixed-dataset evaluation only; group-safe generalization is unavailable."
        ),
        "run_fingerprints": sorted({str(r.get("run_fingerprint", "")) for r in runs}),
        "model": runs[0].get("model"),
        "folds": folds,
        "seeds": seeds,
        "outer_macro_f1": summarize([r["outer_val"]["macro_f1"] for r in runs]),
        "outer_top1": summarize([r["outer_val"]["top1"] for r in runs]),
        "outer_weighted_f1": summarize([r["outer_val"]["weighted_f1"] for r in runs]),
        "per_class_f1_mean": [
            statistics.fmean(r["outer_per_class_f1"][idx] for r in runs)
            for idx in range(len(class_names))
        ] if have_pcf else [],
        "class_names": class_names,
        "runs": [
            {key: r.get(key) for key in ("fold", "seed", "best_epoch", "inner_val", "outer_val")}
            for r in runs
        ],
    }
    out = run_dir / "protocol_summary.json"
    out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    m = aggregate["outer_macro_f1"]
    print(
        f"{run_dir.name}: {len(runs)} runs, "
        f"macro_f1={m['mean']:.4f}+/-{m['std']:.4f}, "
        f"claimable={claimability['claimable']} scope={claimability['claim_scope']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
