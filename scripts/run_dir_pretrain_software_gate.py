#!/usr/bin/env python3
"""Execute the preregistered pre-training software gates for Dir-v1/control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwnas_fpga.fourstage_operator import (  # noqa: E402
    architecture_sha256,
    build_fourstage_architecture,
    parameter_and_mac_count,
)
from hwnas_fpga.fourstage_selection import canonical_sha256  # noqa: E402
from hwnas_fpga.models import (  # noqa: E402
    DirMBConv3Split11E3V1Block,
    SplitDW3ControlBlock,
    build_model,
)
from hwnas_fpga.search_space import ArchitectureSpec  # noqa: E402
from hwnas_fpga.training.protocol_reporting import sha256_file  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction-summary",
        default="artifacts/sonar_fourstage_operator_v2/direction_gate_summary.json",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sonar_fourstage_operator_v2/dir_pretrain_software_gate.json",
    )
    args = parser.parse_args()
    direction_path = Path(args.direction_summary).resolve()
    direction = json.loads(direction_path.read_text(encoding="utf-8"))
    if direction.get("status") != "DIRECTIONAL_BASIS_PASS":
        raise ValueError("Dir implementation is blocked without directional basis")

    torch.manual_seed(20260731)
    dir_block = DirMBConv3Split11E3V1Block(32, 32).train()
    control_block = SplitDW3ControlBlock(32, 32).train()
    random_input = torch.randn(2, 32, 28, 28, requires_grad=True)
    random_output = dir_block(random_input)
    loss = random_output.square().mean()
    loss.backward()
    finite_gradients = bool(
        random_input.grad is not None
        and torch.isfinite(random_input.grad).all()
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in dir_block.parameters()
        )
    )

    boundary_rows = {}
    dir_block.eval()
    with torch.no_grad():
        for name, value in (
            ("zeros", 0.0),
            ("negative_full_scale", -128.0),
            ("positive_full_scale", 127.0),
        ):
            probe = torch.full((1, 32, 28, 28), value)
            observed = dir_block(probe)
            boundary_rows[name] = {
                "finite": bool(torch.isfinite(observed).all()),
                "shape": list(observed.shape),
                "sha256": tensor_sha256(observed),
            }

    residual_probe = torch.randn(1, 32, 28, 28)
    residual_block = DirMBConv3Split11E3V1Block(32, 32).eval()
    with torch.no_grad():
        for parameter in residual_block.parameters():
            parameter.zero_()
        residual_exact = torch.equal(
            residual_block(residual_probe), residual_probe
        )

    dir_arch = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="dir_mbconv3_split11_e3_v1",
    )
    control_arch = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="split_dw3_control",
    )
    k3_arch = build_fourstage_architecture(
        stage2_kernel=5,
        stage2_expansion=6,
        stage4_op="mbconv_k3_e3",
    )
    restored = ArchitectureSpec.from_dict(dir_arch.to_dict())
    roundtrip_exact = restored == dir_arch
    model = build_model(dir_arch, num_classes=8).eval()
    with tempfile.TemporaryDirectory(prefix="dir_v1_gate_") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint)
        reloaded = build_model(restored, num_classes=8).eval()
        reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        serialization_probe = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            serialization_exact = torch.equal(
                model(serialization_probe), reloaded(serialization_probe)
            )
        checkpoint_sha = sha256_file(checkpoint)

    dir_cost = parameter_and_mac_count(dir_arch)
    control_cost = parameter_and_mac_count(control_arch)
    k3_cost = parameter_and_mac_count(k3_arch)
    components = {
        "shape_32x28x28_preserved": list(random_output.shape)
        == [2, 32, 28, 28],
        "forward_backward_finite": finite_gradients,
        "residual_exact_with_zero_main_path": residual_exact,
        "branch_order_fixed": dir_block.branch_order
        == ("dw_1x3", "dw_3x1"),
        "branch_widths_48_48": dir_block.branch_channels == 48,
        "dir_kernels_fixed": (
            dir_block.dw_1x3.kernel_size == (1, 3)
            and dir_block.dw_3x1.kernel_size == (3, 1)
        ),
        "control_kernels_fixed": (
            control_block.dw_3x3_first.kernel_size == (3, 3)
            and control_block.dw_3x3_second.kernel_size == (3, 3)
        ),
        "architecture_roundtrip_exact": roundtrip_exact,
        "architecture_hash_roundtrip_exact": architecture_sha256(restored)
        == architecture_sha256(dir_arch),
        "checkpoint_reload_exact": serialization_exact,
        "control_matches_k3_params": control_cost["parameter_count"]
        == k3_cost["parameter_count"],
        "control_matches_k3_macs": control_cost["macs"] == k3_cost["macs"],
        "dir_has_lower_params_than_control": dir_cost["parameter_count"]
        < control_cost["parameter_count"],
        "dir_has_lower_macs_than_control": dir_cost["macs"]
        < control_cost["macs"],
        "boundary_tensors_finite": all(
            row["finite"] for row in boundary_rows.values()
        ),
    }
    passed = all(components.values())
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "operator": "dir_mbconv3_split11_e3_v1",
        "mechanism_control": "split_dw3_control",
        "placement": {
            "stage": 4,
            "resolution": 28,
            "in_channels": 32,
            "out_channels": 32,
            "stride": 1,
            "expand_ratio": 3,
            "split": [48, 48],
            "branch_order": ["dw_1x3", "dw_3x1"],
        },
        "components": components,
        "boundary_rows": boundary_rows,
        "architecture_sha256": architecture_sha256(dir_arch),
        "checkpoint_roundtrip_sha256": checkpoint_sha,
        "software_cost": {
            "dir": dir_cost,
            "split_dw3_control": control_cost,
            "mbconv_k3_e3": k3_cost,
        },
        "direction_gate": {
            "path": str(direction_path),
            "status": direction["status"],
            "payload_sha256": direction["payload_sha256"],
        },
        "claim_boundary": (
            "This is an executed FP32 graph/serialization/software-cost gate. "
            "It is not INT8 parity, HLS, route, accuracy, robustness, board, "
            "or power evidence."
        ),
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
