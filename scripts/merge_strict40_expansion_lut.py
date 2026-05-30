#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml


SPEC_KEYS = (
    "op",
    "kernel_size",
    "in_channels",
    "out_channels",
    "stride",
    "groups",
    "expand_ratio",
    "input_resolution",
    "bitwidth",
    "input_parallelism",
    "output_parallelism",
    "unroll_factor",
    "target_clock_mhz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge strict40 expansion HLS measurements into a strict formal LUT."
    )
    parser.add_argument("--base-lut", default="hls_lut_builder/results/formal_lut_strict40_v1.json")
    parser.add_argument("--base-status", default="hls_lut_builder/results/formal_lut_status_strict40_v1.json")
    parser.add_argument("--expansion-manifest", default="results/strict40_expansion_hls_manifest_short.yaml")
    parser.add_argument("--base-config", default="configs/search/formal_lut_strict40_nksid_rl50_eval5_explore_av7k325.yaml")
    parser.add_argument(
        "--output-lut",
        default="hls_lut_builder/results/formal_lut_strict40_expanded_v1.json",
    )
    parser.add_argument(
        "--output-status",
        default="hls_lut_builder/results/formal_lut_status_strict40_expanded_v1.json",
    )
    parser.add_argument(
        "--output-config",
        default="configs/search/formal_lut_strict40_expanded_nksid_rl200_eval10_explore_av7k325.yaml",
    )
    parser.add_argument("--report", default="results/strict40_lut_stage2_merge_report.md")
    parser.add_argument("--summary-json", default="results/strict40_lut_stage2_merge_summary.json")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def normalize_search_spec(raw_spec: dict[str, Any]) -> dict[str, Any]:
    spec = {key: raw_spec.get(key) for key in SPEC_KEYS if key in raw_spec}
    spec["input_resolution"] = [int(v) for v in spec.get("input_resolution", [0, 0])]
    for key in (
        "kernel_size",
        "in_channels",
        "out_channels",
        "stride",
        "groups",
        "expand_ratio",
        "bitwidth",
        "input_parallelism",
        "output_parallelism",
        "unroll_factor",
    ):
        spec[key] = int(spec[key])
    # Search-time OpSpec does not include the clock for packed_stream_v1 rows.
    spec["target_clock_mhz"] = None
    return spec


def spec_key(raw_spec: dict[str, Any]) -> str:
    spec = normalize_search_spec(raw_spec)
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def hls_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    payload = entry.get("hls_estimate", {})
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict) or not metrics:
        metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Expansion entry is missing HLS metrics: {entry.get('metadata', {}).get('case_name')}")
    return metrics


def downstream_power(entry: dict[str, Any]) -> tuple[float | None, str]:
    vivado_actual = entry.get("vivado_actual")
    if not isinstance(vivado_actual, dict):
        return None, "missing"
    post_route = vivado_actual.get("metrics", {}).get("post_route", {})
    if not isinstance(post_route, dict):
        return None, "missing"
    power_w = post_route.get("power_w")
    if power_w in (None, ""):
        return None, "missing"
    return float(power_w), str(post_route.get("power_source") or "implementation_vectorless_report")


def entry_from_manifest(raw_entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = raw_entry.get("metadata", {}) if isinstance(raw_entry.get("metadata"), dict) else {}
    spec = normalize_search_spec(raw_entry["op_spec"])
    metrics = hls_metrics(raw_entry)
    power_w, power_source = downstream_power(raw_entry)
    latency_ms = float(metrics.get("latency_ms") or 0.0)
    cycles = int(metrics.get("cycles") or 0)
    bram = int(metrics.get("bram") or metrics.get("bram_18k") or 0)
    lut_entry = {
        "op_spec": spec,
        "latency_ms": latency_ms,
        "cycles": cycles,
        "dsp": int(metrics.get("dsp") or 0),
        "bram": bram,
        "lut": int(metrics.get("lut") or 0),
        "power_w": power_w,
        "energy_mj": None if power_w is None else power_w * latency_ms,
    }
    case_name = str(metadata.get("case_name") or raw_entry.get("case_name") or spec["op"])
    status_entry = {
        "case_name": case_name,
        "status": "measured",
        "op_type": spec["op"],
        "lookup_op": spec["op"],
        "defer_reason": None,
        "root_cause_bucket": None,
        "status_source": "strict40_short_hls_stage1",
        "original_status": None,
        "op_spec": spec,
        "board_cycles": cycles,
        "board_latency_ms": latency_ms,
        "deployable_at_200mhz": True,
        "detail": "hls_synthesis_success",
        "metrics_source": "hls_estimate",
        "power_w": power_w,
        "power_source": power_source,
        "downstream_status": downstream_status(raw_entry),
    }
    return lut_entry, status_entry, {"case_name": case_name, **lut_entry, "power_source": power_source}


def downstream_status(entry: dict[str, Any]) -> str:
    vivado_actual = entry.get("vivado_actual")
    if isinstance(vivado_actual, dict) and vivado_actual:
        return str(vivado_actual.get("status") or "success")
    metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
    return str(metadata.get("downstream_status") or "missing")


def expanded_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["project"]["name"] = "formal-lut-strict40-expanded-nksid-rl200-eval10-explore-av7k325"
    config["project"]["run_name"] = "formal_lut_strict40_expanded_nksid_rl200_eval10_explore_v1"
    config["search_space"]["stage_block_choices"] = [
        [{"op": "conv", "kernel_size": 1, "expand_ratio": 1}],
        [
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6},
        ],
        [
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6},
        ],
        [
            {"op": "skip", "kernel_size": 1, "expand_ratio": 1},
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3},
            {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6},
            {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6},
        ],
    ]
    config["search_space"]["kernel_choices"] = [1, 3, 5]
    config["search_space"]["expand_choices"] = [1, 3, 6]
    config["search_space"]["op_choices"] = ["conv", "mbconv", "skip"]
    config["hardware"]["lut_path"] = "hls_lut_builder/results/formal_lut_strict40_expanded_v1.json"
    config["hardware"]["formal_lut_status_path"] = "hls_lut_builder/results/formal_lut_status_strict40_expanded_v1.json"
    config["search"]["method"] = "rl"
    config["search"]["num_candidates"] = 200
    config["search"]["episodes"] = 200
    config["search"]["eval_epochs"] = 10
    config["search"]["eval_early_stopping_patience"] = 1
    config["search"]["exploration_epsilon_decay_episodes"] = 200
    config["training"]["epochs"] = 1
    return config


def write_report(path: str | Path, *, summary: dict[str, Any], added_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Strict40 Stage 2 LUT merge report\n\n",
        f"Generated: {summary['generated_at']}\n\n",
        "## Outputs\n\n",
        f"- LUT: `{summary['output_lut']}`\n",
        f"- Status: `{summary['output_status']}`\n",
        f"- Expanded search config: `{summary['output_config']}`\n\n",
        "## Merge Summary\n\n",
        f"- Base LUT entries: {summary['base_lut_entries']}\n",
        f"- Base status entries: {summary['base_status_entries']}\n",
        f"- Added measured expansion entries: {summary['added_entries']}\n",
        f"- Output LUT entries: {summary['output_lut_entries']}\n",
        f"- Output status entries: {summary['output_status_entries']}\n",
        f"- Power coverage: {summary['power_coverage']['measured']} measured / {summary['power_coverage']['missing']} missing\n\n",
        "## Added Cases\n\n",
        "| case_name | op | res | in->out | stride | latency_ms | LUT | DSP | BRAM | power_w | power_source |\n",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in added_rows:
        spec = row["op_spec"]
        power = "" if row["power_w"] is None else f"{float(row['power_w']):.3f}"
        lines.append(
            f"| `{row['case_name']}` | `{spec['op']}` | {spec['input_resolution'][0]} | "
            f"{spec['in_channels']}->{spec['out_channels']} | {spec['stride']} | "
            f"{float(row['latency_ms']):.3f} | {int(row['lut'])} | {int(row['dsp'])} | "
            f"{int(row['bram'])} | {power} | {row['power_source']} |\n"
        )
    lines.append(
        "\nPower is intentionally left missing for cases without downstream post-route power; "
        "no imputed power was written into the LUT.\n"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_lut = load_json(args.base_lut)
    base_status = load_json(args.base_status)
    manifest = load_yaml(args.expansion_manifest)
    base_config = load_yaml(args.base_config)

    base_lut_entries = list(base_lut.get("entries", []))
    base_status_entries = list(base_status.get("entries", []))
    expansion_entries = manifest.get("entries", [])
    if not isinstance(expansion_entries, list) or len(expansion_entries) != 7:
        raise ValueError(f"Expected exactly 7 expansion entries, got {len(expansion_entries)}")

    added_lut_entries: list[dict[str, Any]] = []
    added_status_entries: list[dict[str, Any]] = []
    added_rows: list[dict[str, Any]] = []
    for raw_entry in expansion_entries:
        lut_entry, status_entry, row = entry_from_manifest(raw_entry)
        added_lut_entries.append(lut_entry)
        added_status_entries.append(status_entry)
        added_rows.append(row)

    added_keys = {spec_key(entry["op_spec"]) for entry in added_lut_entries}
    kept_lut_entries = [entry for entry in base_lut_entries if spec_key(entry["op_spec"]) not in added_keys]
    kept_status_entries = [entry for entry in base_status_entries if spec_key(entry["op_spec"]) not in added_keys]

    output_lut_entries = kept_lut_entries + added_lut_entries
    output_status_entries = kept_status_entries + added_status_entries
    status_counts = Counter(str(entry["status"]) for entry in output_status_entries)
    power_coverage = {
        "measured": sum(1 for entry in added_lut_entries if entry.get("power_w") is not None),
        "missing": sum(1 for entry in added_lut_entries if entry.get("power_w") is None),
    }
    generated_at = datetime.now().astimezone().isoformat()
    metadata = dict(base_lut.get("metadata") or {})
    metadata.update(
        {
            "generated_at": generated_at,
            "implementation_version": "strict40_expanded_v1_short_hls_2026_05_15",
            "source_base_lut": str(Path(args.base_lut).resolve()),
            "source_base_status": str(Path(args.base_status).resolve()),
            "source_expansion_manifest": str(Path(args.expansion_manifest).resolve()),
            "added_measured_entries": len(added_lut_entries),
            "power_coverage": power_coverage,
            "status_counts": dict(status_counts),
            "query_key_note": "target_clock_mhz is intentionally set to null to match search-time OpSpec keys.",
        }
    )

    output_lut = {"entries": output_lut_entries, "duplicates": {}, "metadata": metadata}
    output_status = {"metadata": metadata, "entries": output_status_entries}
    write_json(args.output_lut, output_lut)
    write_json(args.output_status, output_status)

    config = expanded_config(base_config)
    write_yaml(args.output_config, config)

    summary = {
        "generated_at": generated_at,
        "base_lut": str(Path(args.base_lut).resolve()),
        "base_status": str(Path(args.base_status).resolve()),
        "expansion_manifest": str(Path(args.expansion_manifest).resolve()),
        "output_lut": args.output_lut,
        "output_status": args.output_status,
        "output_config": args.output_config,
        "base_lut_entries": len(base_lut_entries),
        "base_status_entries": len(base_status_entries),
        "added_entries": len(added_lut_entries),
        "output_lut_entries": len(output_lut_entries),
        "output_status_entries": len(output_status_entries),
        "status_counts": dict(status_counts),
        "power_coverage": power_coverage,
        "stage_block_choice_counts": [len(stage) for stage in config["search_space"]["stage_block_choices"]],
    }
    write_json(args.summary_json, summary)
    write_report(args.report, summary=summary, added_rows=added_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
