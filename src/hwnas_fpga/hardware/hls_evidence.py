"""Candidate-level HLS evidence assembly with no analytic fallback."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from hwnas_fpga.hardware import parse_hls_report
from hwnas_fpga.hardware.calibration_v2 import canonical_sha256
from hwnas_fpga.search_space import ArchitectureSpec, SearchSpace


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_op_key(op_spec: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "op": str(op_spec.get("op", "")),
            "kernel_size": int(op_spec.get("kernel_size", 1)),
            "in_channels": int(op_spec.get("in_channels", 1)),
            "out_channels": int(op_spec.get("out_channels", 1)),
            "stride": int(op_spec.get("stride", 1)),
            "groups": int(op_spec.get("groups", 1)),
            "expand_ratio": int(op_spec.get("expand_ratio", 1)),
            "input_resolution": [
                int(value) for value in op_spec.get("input_resolution", [1, 1])
            ],
            "bitwidth": int(op_spec.get("bitwidth", 8)),
            "input_parallelism": int(op_spec.get("input_parallelism", 1)),
            "output_parallelism": int(op_spec.get("output_parallelism", 1)),
            "unroll_factor": int(op_spec.get("unroll_factor", 1)),
        }
    )


def load_status_index(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    index: dict[str, dict[str, Any]] = {}
    for entry in payload.get("entries", []):
        key = normalized_op_key(entry["op_spec"])
        if key in index and index[key].get("case_name") != entry.get("case_name"):
            raise ValueError(f"ambiguous HLS status key {key}")
        index[key] = dict(entry)
    return index


def _op_spec(
    *,
    op: str,
    kernel_size: int,
    in_channels: int,
    out_channels: int,
    stride: int,
    expand_ratio: int,
    input_resolution: int,
) -> dict[str, Any]:
    return {
        "op": op,
        "kernel_size": int(kernel_size),
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "stride": int(stride),
        "groups": 1,
        "expand_ratio": int(expand_ratio),
        "input_resolution": [int(input_resolution), int(input_resolution)],
        "bitwidth": 8,
        "input_parallelism": 1,
        "output_parallelism": 1,
        "unroll_factor": 1,
        "target_clock_mhz": None,
    }


def candidate_roles(
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = [
        {
            "role": "stem",
            "logical_op": "stem_conv",
            "required_hls": True,
            "op_spec": _op_spec(
                op="conv",
                kernel_size=3,
                in_channels=architecture.input_channels,
                out_channels=architecture.stem_channels,
                stride=architecture.stem_stride,
                expand_ratio=1,
                input_resolution=search_space.config.image_size,
            ),
        }
    ]
    resolved = search_space.resolve_blocks(architecture)
    for block in resolved:
        if block.op == "mbconv":
            lookup_op = f"mbconv_e{block.expand_ratio}_k{block.kernel_size}"
        elif block.op in {"conv", "dw_pw_conv"}:
            lookup_op = "conv"
        else:
            lookup_op = block.op
        roles.append(
            {
                "role": f"stage{block.stage_index}_block{block.block_index}",
                "logical_op": block.op,
                "required_hls": block.op != "skip",
                "op_spec": _op_spec(
                    op=lookup_op,
                    kernel_size=block.kernel_size,
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    stride=block.stride,
                    expand_ratio=block.expand_ratio,
                    input_resolution=block.input_resolution,
                ),
            }
        )

    if resolved:
        final_resolution = resolved[-1].output_resolution
        final_channels = resolved[-1].out_channels
    else:
        final_resolution = max(
            1,
            math.ceil(
                search_space.config.image_size
                / architecture.stem_stride
                / architecture.post_stem_downsample_stride
            ),
        )
        final_channels = architecture.stem_channels
    head_channels = architecture.head_conv_channels or architecture.head_channels
    classifier_channels = final_channels
    if head_channels:
        roles.append(
            {
                "role": "head_conv",
                "logical_op": "head_conv",
                "required_hls": True,
                "op_spec": _op_spec(
                    op="conv",
                    kernel_size=1,
                    in_channels=final_channels,
                    out_channels=head_channels,
                    stride=1,
                    expand_ratio=1,
                    input_resolution=final_resolution,
                ),
            }
        )
        classifier_channels = head_channels
    roles.extend(
        [
            {
                "role": "global_avg_pool",
                "logical_op": "global_avg_pool",
                "required_hls": True,
                "op_spec": _op_spec(
                    op="global_avg_pool",
                    kernel_size=final_resolution,
                    in_channels=classifier_channels,
                    out_channels=classifier_channels,
                    stride=1,
                    expand_ratio=1,
                    input_resolution=final_resolution,
                ),
            },
            {
                "role": "classifier",
                "logical_op": "fc_layer",
                "required_hls": True,
                "op_spec": _op_spec(
                    op="fc_layer",
                    kernel_size=1,
                    in_channels=classifier_channels,
                    out_channels=architecture.num_classes or 8,
                    stride=1,
                    expand_ratio=1,
                    input_resolution=1,
                ),
            },
        ]
    )
    return roles


def find_case_report(repo_root: Path, case_name: str) -> Path | None:
    roots = [
        repo_root / "hls_lut_builder" / "results" / "p" / case_name,
        repo_root / "hls_lut_builder" / "results" / "strict40_expansion_p" / case_name,
        repo_root / "hls_lut_builder" / "results" / "candidate_hls_cache" / case_name,
    ]
    for root in roots:
        candidates = sorted(root.glob("project/*/syn/report/csynth.xml"))
        if candidates:
            return candidates[0]
        candidates = sorted(root.glob("project/*/syn/report/*_csynth.xml"))
        if candidates:
            return candidates[0]
    return None


def hls_metrics(path: Path, *, clock_mhz: float = 200.0) -> dict[str, Any]:
    parsed = parse_hls_report(path)
    cycles = int(parsed["cycles"])
    bram18 = float(parsed.get("bram_18k", parsed.get("bram", 0)))
    return {
        "cycles": cycles,
        "latency_ms": cycles / (float(clock_mhz) * 1000.0),
        "dsp": int(parsed["dsp"]),
        "lut": int(parsed["lut"]),
        "bram18": bram18,
        "bram": bram18 / 2.0,
        "estimated_clock_period_ns": parsed.get("estimated_clock_period_ns"),
        "fmax_est_mhz": parsed.get("fmax_est_mhz"),
    }


def assemble_candidate_hls_report(
    *,
    candidate_path: str | Path,
    architecture: ArchitectureSpec,
    search_space: SearchSpace,
    status_index: Mapping[str, Mapping[str, Any]],
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root)
    roles = candidate_roles(architecture, search_space)
    resolved_roles = []
    missing = []
    totals = {
        "cycles": 0,
        "latency_ms": 0.0,
        "dsp": 0,
        "lut": 0,
        "bram": 0.0,
    }
    for role in roles:
        op_spec = role["op_spec"]
        key = normalized_op_key(op_spec)
        status = status_index.get(key)
        resolved = dict(role)
        resolved["op_key"] = key
        if role["logical_op"] == "skip":
            resolved.update(
                {
                    "evidence_status": "structural_skip",
                    "evidence_source": "structural_identity",
                    "hls_metrics": {
                        "cycles": 0,
                        "latency_ms": 0.0,
                        "dsp": 0,
                        "lut": 0,
                        "bram": 0.0,
                    },
                }
            )
            resolved_roles.append(resolved)
            continue
        case_name = str(status.get("case_name")) if status else ""
        report_path = find_case_report(root, case_name) if case_name else None
        if status is None or status.get("status") != "measured" or report_path is None:
            reason = (
                "no_status_entry"
                if status is None
                else f"status_{status.get('status')}"
                if status.get("status") != "measured"
                else "missing_csynth_report"
            )
            resolved.update(
                {
                    "evidence_status": "missing",
                    "missing_reason": reason,
                    "case_name": case_name or None,
                }
            )
            missing.append(
                {
                    "role": role["role"],
                    "reason": reason,
                    "op_key": key,
                    "op_spec": op_spec,
                }
            )
            resolved_roles.append(resolved)
            continue
        metrics = hls_metrics(report_path)
        for metric in totals:
            totals[metric] += metrics[metric]
        resolved.update(
            {
                "evidence_status": "measured_csynth",
                "evidence_source": "hls_estimate",
                "case_name": case_name,
                "report_path": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "hls_metrics": metrics,
            }
        )
        resolved_roles.append(resolved)

    candidate_file = Path(candidate_path)
    return {
        "schema_version": 1,
        "candidate_path": str(candidate_file.resolve()),
        "candidate_sha256": sha256_file(candidate_file),
        "architecture_sha256": canonical_sha256(architecture.to_dict()),
        "evidence_source": "hls_estimate",
        "claim_boundary": (
            "HLS csynth estimates are not post-route resources or COM5 latency."
        ),
        "evidence_complete": len(missing) == 0,
        "required_role_count": sum(role["required_hls"] for role in roles),
        "measured_role_count": sum(
            role.get("evidence_status") == "measured_csynth"
            for role in resolved_roles
        ),
        "aggregate_hls": totals if not missing else None,
        "roles": resolved_roles,
        "missing_cases": missing,
    }

