#!/usr/bin/env python3
"""Generate a paper-ready macro backbone comparison table from a baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_bool(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _normalize_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary:
        rows.append(
            {
                "arch_id": str(row.get("arch_id", "")),
                "display_name": str(row.get("display_name", row.get("arch_id", ""))),
                "backbone": str(row.get("backbone", "")),
                "pretrained_loaded": bool(row.get("pretrained_loaded", False)),
                "params": int(row.get("params", 0)),
                "macs": int(row.get("macs", 0)),
                "top1": float(row.get("top1", 0.0)),
                "top5": float(row.get("top5", 0.0)),
                "macro_f1": float(row.get("macro_f1", 0.0)),
                "weighted_f1": float(row.get("weighted_f1", 0.0)),
                "cpu_latency_ms": float(row.get("cpu_latency_ms", 0.0)),
                "fpga_latency_ms": float(row.get("fpga_latency_ms", 0.0)),
                "peak_dsp": int(row.get("peak_dsp", 0)),
                "peak_bram": int(row.get("peak_bram", 0)),
                "peak_lut": int(row.get("peak_lut", 0)),
                "power_w": float(row.get("power_w", 0.0)),
                "feasible": bool(row.get("feasible", False)),
                "violations": str(row.get("violations", "")),
            }
        )
    return rows


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not row["feasible"],
            -row["macro_f1"],
            -row["top1"],
            row["fpga_latency_ms"],
        ),
    )


def _role_lookup(selected_pool: dict[str, Any]) -> dict[str, str]:
    roles = selected_pool.get("roles", {}) if isinstance(selected_pool, dict) else {}
    lookup: dict[str, str] = {}
    for role_name, payload in roles.items():
        if isinstance(payload, dict):
            arch_id = str(payload.get("arch_id", "")).strip()
            if arch_id:
                lookup[arch_id] = role_name
    return lookup


def render_markdown(
    *,
    run_dir: Path,
    rows: list[dict[str, Any]],
    selected_pool: dict[str, Any] | None,
    run_info: dict[str, Any] | None,
) -> str:
    role_lookup = _role_lookup(selected_pool or {})
    lines = [
        "# Fair Macro Backbone Comparison",
        "",
        f"Run directory: `{run_dir}`",
        "",
    ]
    if isinstance(run_info, dict):
        lines.extend(
            [
                f"- Run name: `{run_info.get('run_name', run_dir.name)}`",
                f"- Status: `{run_info.get('status', 'unknown')}`",
                f"- Dataset: `{run_info.get('dataset_name', 'unknown')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Backbone Table",
            "",
            "| Rank | Arch ID | Display Name | Role | Feasible | Macro-F1 | Top-1 | Top-5 | FPGA Latency (ms) | DSP | BRAM | LUT | Power (W) | Params | MACs | Violations |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for index, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"`{row['arch_id']}`",
                    row["display_name"],
                    role_lookup.get(row["arch_id"], "-"),
                    _fmt_bool(row["feasible"]),
                    _fmt_float(row["macro_f1"]),
                    _fmt_float(row["top1"]),
                    _fmt_float(row["top5"]),
                    _fmt_float(row["fpga_latency_ms"]),
                    str(row["peak_dsp"]),
                    str(row["peak_bram"]),
                    str(row["peak_lut"]),
                    _fmt_float(row["power_w"]),
                    str(row["params"]),
                    str(row["macs"]),
                    row["violations"] or "-",
                ]
            )
            + " |"
        )

    if isinstance(selected_pool, dict):
        lines.extend(
            [
                "",
                "## Selected Anchors",
                "",
                "| Role | Arch ID | Display Name | Backbone | Macro-F1 | FPGA Latency (ms) |",
                "| --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        roles = selected_pool.get("roles", {})
        if isinstance(roles, dict):
            for role_name in ("accuracy_anchor", "search_anchor", "lightweight_anchor"):
                payload = roles.get(role_name)
                if not isinstance(payload, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            role_name,
                            f"`{payload.get('arch_id', '-')}`",
                            str(payload.get("display_name", "-")),
                            str(payload.get("backbone", "-")),
                            _fmt_float(payload.get("macro_f1")),
                            _fmt_float(payload.get("fpga_latency_ms")),
                        ]
                    )
                    + " |"
                )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a markdown summary for one backbone baseline run")
    parser.add_argument("run_dir", type=str, help="Backbone baseline run directory")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output markdown path. Defaults to <run_dir>/results/fair_macro_backbone_table.md",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    summary_path = run_dir / "results" / "backbone_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing backbone summary: {summary_path}")

    summary = _read_json(summary_path)
    if not isinstance(summary, list):
        raise ValueError(f"Expected a list in {summary_path}")

    rows = _sort_rows(_normalize_rows(summary))
    selected_pool_path = run_dir / "results" / "selected_backbone_pool.json"
    run_info_path = run_dir / "run_info.json"
    selected_pool = _read_json(selected_pool_path) if selected_pool_path.exists() else None
    run_info = _read_json(run_info_path) if run_info_path.exists() else None

    markdown = render_markdown(
        run_dir=run_dir,
        rows=rows,
        selected_pool=selected_pool if isinstance(selected_pool, dict) else None,
        run_info=run_info if isinstance(run_info, dict) else None,
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results" / "fair_macro_backbone_table.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown + "\n", encoding="utf-8")
    print(f"[OK] Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
