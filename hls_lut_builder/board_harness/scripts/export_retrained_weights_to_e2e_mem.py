#!/usr/bin/env python3
"""Export retrained HW-NAS checkpoint weights into an e2e board harness.

The existing full-network harness generates deterministic latency-only
parameters.  This script replaces those per-component parameter memories with
BN-folded checkpoint weights while keeping the harness project isolated from
previous Phase0 v3 board evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

import current84_fullcombo_board_validation as fullcombo  # noqa: E402
import generate_harness as gh  # noqa: E402
import sonar_classifier_e2e_variant_build as variant  # noqa: E402


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class RoleExport:
    role: str
    kind: str
    source_tensor: str
    source_shape: list[int]
    quant_shape: list[int]
    scale: float
    data_width: int
    files: list[dict[str, Any]]
    padded_values: int


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def twos_complement_hex(value: int, data_width: int) -> str:
    mask = (1 << data_width) - 1
    hex_digits = max(1, math.ceil(data_width / 4))
    return f"{value & mask:0{hex_digits}x}"


def write_mem(path: Path, values: list[int], data_width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [twos_complement_hex(int(value), data_width) for value in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quantize_symmetric(tensor: torch.Tensor, *, bit_width: int = 8) -> tuple[torch.Tensor, float]:
    qmax = (1 << (bit_width - 1)) - 1
    max_abs = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
    scale = max_abs / qmax if max_abs > 0.0 else 1.0
    quant = torch.clamp(torch.round(tensor / scale), -qmax, qmax).to(torch.int32)
    return quant, float(scale)


def quantize_bias(bias: torch.Tensor, weight_scale: float) -> torch.Tensor:
    scale = weight_scale if weight_scale > 0.0 else 1.0
    q = torch.round(bias / scale).to(torch.int64)
    q = torch.clamp(q, -(1 << 31), (1 << 31) - 1)
    return q.to(torch.int32)


def candidate_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(path)
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else payload
    if not isinstance(candidate, dict):
        raise ValueError(f"Unsupported candidate payload: {path}")
    architecture = candidate.get("encoding") or candidate.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"Candidate has no architecture/encoding dict: {path}")
    return candidate, architecture


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dict: {path}")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")
    tensor_state = {str(key): value.detach().cpu() for key, value in state.items() if torch.is_tensor(value)}
    return payload, tensor_state


def fold_conv_bn(
    state: dict[str, torch.Tensor],
    conv_prefix: str,
    bn_prefix: str,
    consumed: set[str],
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key = f"{conv_prefix}.weight"
    if weight_key not in state:
        raise KeyError(weight_key)
    weight = state[weight_key].float()
    consumed.add(weight_key)
    bias_key = f"{conv_prefix}.bias"
    if bias_key in state:
        bias = state[bias_key].float()
        consumed.add(bias_key)
    else:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)

    gamma = state[f"{bn_prefix}.weight"].float()
    beta = state[f"{bn_prefix}.bias"].float()
    mean = state[f"{bn_prefix}.running_mean"].float()
    var = state[f"{bn_prefix}.running_var"].float()
    consumed.update(
        {
            f"{bn_prefix}.weight",
            f"{bn_prefix}.bias",
            f"{bn_prefix}.running_mean",
            f"{bn_prefix}.running_var",
        }
    )
    tracked = f"{bn_prefix}.num_batches_tracked"
    if tracked in state:
        consumed.add(tracked)

    inv_std = gamma / torch.sqrt(var + eps)
    folded_weight = weight * inv_std.reshape(-1, 1, 1, 1)
    folded_bias = (bias - mean) * inv_std + beta
    return folded_weight, folded_bias


def linear_params(
    state: dict[str, torch.Tensor],
    prefix: str,
    consumed: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"
    if weight_key not in state or bias_key not in state:
        raise KeyError(f"{prefix}.weight/.bias")
    consumed.update({weight_key, bias_key})
    return state[weight_key].float(), state[bias_key].float()


def role_param_prefix(role: str) -> tuple[str, str | None]:
    if role == "l1_stem_direct":
        return "stem.conv.conv", "stem.conv.bn"
    if role == "l2_stage0_block0_direct":
        return "stages.0.0.conv", "stages.0.0.bn"
    if role.startswith("l3_stage1_block0_"):
        base = "stages.1.0"
    elif role.startswith("l4_stage2_block0_"):
        base = "stages.2.0"
    elif role.startswith("l5_stage3_block0_"):
        base = "stages.3.0"
    else:
        base = ""
    if base:
        if role.endswith("_pw_expand"):
            return f"{base}.expand_conv", f"{base}.expand_bn"
        if role.endswith("_dw"):
            return f"{base}.dw_conv", f"{base}.dw_bn"
        if role.endswith("_pw_project"):
            return f"{base}.project_conv", f"{base}.project_bn"
    if role == "l7_fc_classifier_fc":
        return "head.fc", None
    raise KeyError(f"No checkpoint mapping for harness role: {role}")


def apply_activation(values: torch.Tensor, *, relu: bool) -> torch.Tensor:
    if relu:
        return torch.clamp(values, 0, 6).to(torch.int32)
    return torch.clamp(values, -128, 127).to(torch.int32)


def role_uses_relu(role: str) -> bool:
    return (
        role == "l1_stem_direct"
        or role == "l2_stage0_block0_direct"
        or role.endswith("_pw_expand")
        or role.endswith("_dw")
    )


def tensor_for_hls(role: str, weight: torch.Tensor) -> torch.Tensor:
    if role.endswith("_dw"):
        if weight.ndim != 4 or weight.shape[1] != 1:
            raise ValueError(f"Depthwise weight must be [C,1,K,K], got {tuple(weight.shape)} for {role}")
        return weight[:, 0, :, :].contiguous()
    if role in {"l2_stage0_block0_direct", "l7_fc_classifier_fc"} or role.endswith("_pw_expand") or role.endswith("_pw_project"):
        return weight.reshape(weight.shape[0], weight.shape[1], -1).squeeze(-1).contiguous()
    return weight.contiguous()


def build_role_tensors(
    manifest: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> tuple[dict[str, dict[str, torch.Tensor]], set[str]]:
    consumed: set[str] = set()
    role_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for component in manifest.get("components", []):
        role = str(component.get("role", ""))
        if role == "l6_global_avg_pool_gap":
            continue
        conv_or_linear, bn = role_param_prefix(role)
        if bn is None:
            weight, bias = linear_params(state, conv_or_linear, consumed)
        else:
            weight, bias = fold_conv_bn(state, conv_or_linear, bn, consumed)
        role_tensors[role] = {
            "weight_fp32": tensor_for_hls(role, weight),
            "bias_fp32": bias.contiguous(),
        }
    return role_tensors, consumed


def resolve_case_dir(case_name: str) -> Path:
    return variant.resolve_case_dir(case_name)


def build_component_specs(manifest: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for component in manifest.get("components", []):
        case_name = str(component["case_name"])
        role = str(component["role"])
        spec = fullcombo.build_component_spec(role, case_name, resolve_case_dir(case_name), project_root / "export_rtl")
        spec.update(component)
        specs.append(spec)
    return specs


def flatten_outer(tensor: torch.Tensor) -> tuple[list[list[int]], int]:
    if tensor.ndim == 0:
        rows = [[int(tensor.item())]]
        return rows, 1
    outer = int(tensor.shape[0])
    rows = tensor.reshape(outer, -1).to(torch.int64).tolist()
    return [[int(v) for v in row] for row in rows], outer


def write_tensor_mem(
    *,
    data_dir: Path,
    role: str,
    kind: str,
    tensor: torch.Tensor,
    spec: dict[str, Any],
) -> RoleExport:
    iface = spec["param_interfaces"][kind]
    case_meta = spec["case_meta"]
    constants = spec["constants"]
    rows, outer = flatten_outer(tensor)
    flat = [value for row in rows for value in row]
    files: list[dict[str, Any]] = []
    padded_values = 0
    data_width = int(iface.get("data_width", 1))

    if iface["mode"] == "absent":
        return RoleExport(role, kind, "", list(tensor.shape), list(tensor.shape), 1.0, data_width, files, 0)
    if iface["mode"] == "scalar":
        return RoleExport(role, kind, "ignored_scalar", list(tensor.shape), list(tensor.shape), 1.0, data_width, files, 0)
    if iface["mode"] == "bram":
        depth = int(case_meta[f"{kind}_depth"])
        if len(flat) > depth:
            raise ValueError(f"{role} {kind} has {len(flat)} values but BRAM depth is {depth}")
        values = flat + [0] * (depth - len(flat))
        padded_values += depth - len(flat)
        mem_path = data_dir / f"{role}_{kind}.mem"
        write_mem(mem_path, values, data_width)
        files.append(
            {
                "mem_file": str(mem_path),
                "depth": depth,
                "value_count": len(flat),
                "padded_values": depth - len(flat),
                "sha256": sha256_file(mem_path),
            }
        )
    elif iface["mode"] == "partitioned_bram":
        banks = list(iface.get("banks", []))
        if not banks:
            raise ValueError(f"{role} {kind} partitioned interface has no banks")
        partition_counts = gh.bank_partition_counts(banks)
        bank_depth = gh.partitioned_bank_depth(kind, constants, banks, int(case_meta[f"{kind}_depth"]))
        for bank in banks:
            bank_indices = tuple(int(idx) for idx in bank.get("indices", []))
            bank_rows: list[int] = []
            if len(bank_indices) == 1:
                bank_count = int(partition_counts[0]) if partition_counts else len(banks)
                for outer_idx in range(outer):
                    if outer_idx % bank_count == bank_indices[0]:
                        bank_rows.extend(rows[outer_idx])
            elif len(bank_indices) == 2:
                if tensor.ndim < 2:
                    raise ValueError(f"{role} {kind} two-dimensional bank requires tensor.ndim >= 2")
                out_count, in_count = int(partition_counts[0]), int(partition_counts[1])
                shaped = tensor.to(torch.int64)
                for out_idx in range(int(shaped.shape[0])):
                    if out_idx % out_count != bank_indices[0]:
                        continue
                    for in_idx in range(int(shaped.shape[1])):
                        if in_idx % in_count != bank_indices[1]:
                            continue
                        bank_rows.extend(int(v) for v in shaped[out_idx, in_idx].reshape(-1).tolist())
            else:
                raise ValueError(f"{role} {kind} supports only 1-D or 2-D partitioned banks, got {bank_indices}")
            if len(bank_rows) > bank_depth:
                raise ValueError(
                    f"{role} {kind} bank {bank['prefix']} has {len(bank_rows)} values but depth is {bank_depth}"
                )
            values = bank_rows + [0] * (bank_depth - len(bank_rows))
            padded_values += bank_depth - len(bank_rows)
            mem_path = data_dir / f"{role}_{bank['prefix']}.mem"
            write_mem(mem_path, values, int(bank["data_width"]))
            files.append(
                {
                    "mem_file": str(mem_path),
                    "bank_prefix": bank["prefix"],
                    "bank_indices": list(bank.get("indices", [])),
                    "depth": bank_depth,
                    "value_count": len(bank_rows),
                    "padded_values": bank_depth - len(bank_rows),
                    "sha256": sha256_file(mem_path),
                }
            )
    else:
        raise ValueError(f"Unsupported {role} {kind} interface mode: {iface['mode']}")

    return RoleExport(role, kind, "", list(tensor.shape), list(tensor.shape), 1.0, data_width, files, padded_values)


def read_input_tensor(project_root: Path) -> torch.Tensor:
    input_mem = project_root / "data" / "input_stream_words.mem"
    if not input_mem.exists():
        raise FileNotFoundError(input_mem)
    values: list[int] = []
    for line in input_mem.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        value = int(raw, 16)
        if value >= 128:
            value -= 256
        values.append(value)
    if len(values) != 224 * 224:
        raise ValueError(f"Expected 50176 input words for 224x224x1, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 224, 224)


def checksum_for_output(output: torch.Tensor) -> tuple[int, int, int]:
    values = [int(v) for v in output.reshape(-1).tolist()]
    if len(values) != 8:
        raise ValueError(f"Expected 8 classifier output lanes, got {len(values)}")
    bytes_ = [value & 0xFF for value in values]
    word0 = bytes_[0] | (bytes_[1] << 8) | (bytes_[2] << 16) | (bytes_[3] << 24)
    word1 = bytes_[4] | (bytes_[5] << 8) | (bytes_[6] << 16) | (bytes_[7] << 24)
    checksum = (word0 ^ word1) & 0xFFFFFFFF
    signed = [value if value < 128 else value - 256 for value in bytes_]
    argmax = max(range(len(signed)), key=lambda idx: signed[idx])
    packed = ((argmax & 0xFF) << 24) | (checksum & 0xFFFFFF)
    return checksum, argmax, packed


def software_forward(
    project_root: Path,
    quantized_roles: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    x = read_input_tensor(project_root)

    def conv(role: str, stride: int, padding: int, groups: int = 1) -> torch.Tensor:
        params = quantized_roles[role]
        weight = params["weight_q"].float()
        if weight.ndim == 2:
            weight = weight[:, :, None, None]
        elif weight.ndim == 3:
            weight = weight[:, None, :, :]
        y = F.conv2d(
            x.float(),
            weight,
            params["bias_q"].float(),
            stride=stride,
            padding=padding,
            groups=groups,
        )
        return apply_activation(torch.round(y), relu=role_uses_relu(role)).float()

    x = conv("l1_stem_direct", stride=2, padding=1)
    x = conv("l2_stage0_block0_direct", stride=1, padding=0)
    x = conv("l3_stage1_block0_pw_expand", stride=1, padding=0)
    x = conv("l3_stage1_block0_dw", stride=2, padding=1, groups=int(x.shape[1]))
    x = conv("l3_stage1_block0_pw_project", stride=1, padding=0)
    x = conv("l4_stage2_block0_pw_expand", stride=1, padding=0)
    x = conv("l4_stage2_block0_dw", stride=2, padding=1, groups=int(x.shape[1]))
    x = conv("l4_stage2_block0_pw_project", stride=1, padding=0)
    x = conv("l5_stage3_block0_pw_expand", stride=1, padding=0)
    x = conv("l5_stage3_block0_dw", stride=1, padding=1, groups=int(x.shape[1]))
    x = conv("l5_stage3_block0_pw_project", stride=1, padding=0)

    sums = x.to(torch.int32).sum(dim=(2, 3))
    x = torch.div(sums, 28 * 28, rounding_mode="trunc").clamp(-128, 127).float()

    fc = quantized_roles["l7_fc_classifier_fc"]
    logits = F.linear(x.float(), fc["weight_q"].float(), fc["bias_q"].float())
    output = apply_activation(torch.round(logits), relu=False).to(torch.int32)
    checksum, argmax, packed = checksum_for_output(output)
    return {
        "status": "computed_integer_hls_proxy",
        "scope": "deterministic harness input only; not NKSID validation accuracy",
        "activation_rules": {
            "relu_layers": "relu6_quant clamps integer accumulator to [0, 6]",
            "project_fc_layers": "saturate_to_data clamps integer accumulator to int8 range",
            "gap": "integer sum divided by 28*28 with truncation toward zero",
        },
        "output_int8": [int(v) for v in output.reshape(-1).tolist()],
        "checksum32": checksum,
        "checksum24": checksum & 0xFFFFFF,
        "argmax": argmax,
        "checksum_argmax_packed": packed,
    }


def export_to_project(
    *,
    checkpoint: Path,
    candidate_path: Path,
    harness_manifest: Path,
    output_manifest: Path | None = None,
) -> dict[str, Any]:
    project_root = harness_manifest.parent
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    manifest = read_json(harness_manifest)
    candidate, architecture = candidate_payload(candidate_path)
    checkpoint_payload, state = load_checkpoint(checkpoint)
    checkpoint_arch = checkpoint_payload.get("architecture")
    if isinstance(checkpoint_arch, dict) and canonical_json(checkpoint_arch) != canonical_json(architecture):
        raise ValueError("Candidate architecture does not match checkpoint architecture")
    source_candidate = checkpoint_payload.get("source_candidate")
    if isinstance(source_candidate, dict) and source_candidate.get("arch_id") != candidate.get("arch_id"):
        raise ValueError(
            f"Candidate arch_id {candidate.get('arch_id')} does not match checkpoint source "
            f"{source_candidate.get('arch_id')}"
        )

    role_tensors, consumed = build_role_tensors(manifest, state)
    unconsumed = sorted(set(state) - consumed)
    if unconsumed:
        raise ValueError(f"Unconsumed checkpoint state_dict tensors: {unconsumed}")

    specs = build_component_specs(manifest, project_root)
    specs_by_role = {spec["role"]: spec for spec in specs}
    role_exports: list[dict[str, Any]] = []
    quantized_roles: dict[str, dict[str, torch.Tensor]] = {}

    for role, tensors in role_tensors.items():
        spec = specs_by_role[role]
        weight_q, weight_scale = quantize_symmetric(tensors["weight_fp32"], bit_width=8)
        bias_q = quantize_bias(tensors["bias_fp32"], weight_scale)
        quantized_roles[role] = {
            "weight_q": weight_q.reshape_as(tensors["weight_fp32"]).to(torch.int32),
            "bias_q": bias_q.to(torch.int32),
        }

        weight_export = write_tensor_mem(
            data_dir=data_dir,
            role=role,
            kind="weights",
            tensor=weight_q.reshape_as(tensors["weight_fp32"]),
            spec=spec,
        )
        bias_export = write_tensor_mem(
            data_dir=data_dir,
            role=role,
            kind="biases",
            tensor=bias_q,
            spec=spec,
        )
        weight_export.source_tensor = "BN-folded weight" if role != "l7_fc_classifier_fc" else "linear weight"
        weight_export.scale = weight_scale
        bias_export.source_tensor = "BN-folded bias" if role != "l7_fc_classifier_fc" else "linear bias"
        bias_export.scale = weight_scale
        role_exports.append({"role": role, "weights": weight_export.__dict__, "biases": bias_export.__dict__})

    parity = software_forward(project_root, quantized_roles)
    parity_path = project_root / "software_board_parity.json"
    write_json(parity_path, parity)

    out_manifest = output_manifest or (project_root / "trained_weight_manifest.json")
    payload = {
        "generated_at": timestamp(),
        "parameter_mode": "trained_checkpoint_mem",
        "scope": "trained checkpoint weights exported into full-network board harness parameter memories",
        "accuracy_claim": "none",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "metrics": checkpoint_payload.get("metrics", {}),
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": sha256_file(candidate_path),
            "arch_id": candidate.get("arch_id", ""),
            "metrics": candidate.get("metrics", {}),
        },
        "harness": {
            "manifest": str(harness_manifest.resolve()),
            "project_root": str(project_root.resolve()),
            "case_name": manifest.get("case_name", ""),
            "run_name": manifest.get("run_name") or manifest.get("variant_name", ""),
        },
        "quantization": {
            "weight": "per-role symmetric int8",
            "bias": "round(BN-folded fp32 bias / weight_scale) to int32",
            "zero_point": 0,
            "input_scale_assumed": 1.0,
            "batch_norm": "folded into preceding Conv2d weights and biases with eps=1e-5",
        },
        "exports": role_exports,
        "software_board_parity": str(parity_path.resolve()),
    }
    write_json(out_manifest, payload)

    notes = list(manifest.get("notes", [])) if isinstance(manifest.get("notes", []), list) else []
    notes.append(
        "Retrained checkpoint parameters were exported into harness .mem files; COM5 checksum/argmax is a deterministic harness sanity result, not validation-set accuracy."
    )
    manifest.update(
        {
            "parameter_mode": "trained_checkpoint_mem",
            "accuracy_claim": "none",
            "trained_weight_manifest": str(out_manifest.resolve()),
            "software_board_parity": str(parity_path.resolve()),
            "trained_weight_source": payload["checkpoint"],
            "notes": notes,
        }
    )
    write_json(harness_manifest, manifest)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export retrained checkpoint weights into an e2e harness project.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--harness-manifest", required=True)
    parser.add_argument("--output-manifest", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = export_to_project(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        candidate_path=Path(args.candidate_path).expanduser().resolve(),
        harness_manifest=Path(args.harness_manifest).expanduser().resolve(),
        output_manifest=Path(args.output_manifest).expanduser().resolve() if args.output_manifest else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
