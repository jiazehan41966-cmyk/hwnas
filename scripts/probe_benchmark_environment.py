#!/usr/bin/env python3
"""Run one paper-specific import/contract probe inside a dedicated runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _git_commit(checkout: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--probe", required=True, choices=("hwpr", "sure", "harp", "esda", "dmcl", "plud"))
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--pinned-commit", required=True)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    checkout = (PROJECT_ROOT / args.checkout).resolve()
    observed_commit = _git_commit(checkout)
    if observed_commit != args.pinned_commit:
        raise RuntimeError(f"checkout pin mismatch: {observed_commit} != {args.pinned_commit}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the dedicated runtime")
    cuda_sum = float(torch.ones(4, device="cuda").sum().item())
    details: dict[str, object]
    if args.probe == "sure":
        from hwnas_fpga.benchmarks.sure import load_sure_author_components

        components = load_sure_author_components(
            checkout, pinned_commit=args.pinned_commit
        )
        details = {
            "component": "SURE author FMFP/CRL/SAM source loader",
            "source_hashes": components.source_hashes,
        }
    elif args.probe == "dmcl":
        from hwnas_fpga.benchmarks.dmcl import load_dmcl_author_components

        if not args.archive_sha256:
            raise ValueError("DMCL probe requires --archive-sha256")
        components = load_dmcl_author_components(
            checkout,
            pinned_commit=args.pinned_commit,
            archive_sha256=args.archive_sha256,
        )
        details = {
            "component": "author DynamicMarginLoss loader",
            "source_hashes": components.source_hashes,
        }
    elif args.probe == "plud":
        from hwnas_fpga.benchmarks.plud import load_plud_author_components

        if not args.archive_sha256:
            raise ValueError("PLUD probe requires --archive-sha256")
        components = load_plud_author_components(
            checkout,
            pinned_commit=args.pinned_commit,
            archive_sha256=args.archive_sha256,
        )
        details = {
            "component": "author push_logit_loss loader",
            "source_hashes": components.source_hashes,
        }
    elif args.probe == "harp":
        from hwnas_fpga.benchmarks.harp import HARP_GRAPH_TYPE, inspect_harp_gexf

        graph = (
            checkout
            / "dse_database/generated_graphs/machsuite/processed"
            / HARP_GRAPH_TYPE
            / "aes_processed_result.gexf"
        )
        audit = inspect_harp_gexf(graph)
        if not audit["contract_valid"]:
            raise RuntimeError(f"HARP author graph contract failed: {audit['blockers']}")
        details = {
            "component": "HARP author LLVM program-graph contract",
            "graph_sha256": audit["sha256"],
        }
    elif args.probe == "esda":
        from hwnas_fpga.benchmarks.esda import build_esda_evidence_chain_manifest

        manifest = build_esda_evidence_chain_manifest(checkout)
        if not manifest["contract_valid"]:
            raise RuntimeError(f"ESDA artifact contract failed: {manifest['blockers']}")
        details = {
            "component": "ESDA author artifact evidence-chain contract",
            "evidence_layer_count": len(manifest["evidence_layers"]),
            "cross_platform_numeric_ranking_allowed": False,
        }
    else:
        from hwnas_fpga.benchmarks.hwpr import listmle_pareto_loss

        value = listmle_pareto_loss(
            torch.tensor([0.2, 0.1], device="cuda"),
            torch.tensor([0, 1], device="cuda"),
        )
        details = {
            "component": "project paper-spec ListMLE bridge only",
            "loss_finite": bool(torch.isfinite(value).item()),
            "author_method_runtime_complete": False,
        }
    payload = {
        "paper_id": args.paper_id,
        "probe": args.probe,
        "interpreter": sys.executable,
        "sys_prefix": sys.prefix,
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_tensor_sum": cuda_sum,
        "checkout": str(checkout),
        "observed_commit": observed_commit,
        "details": details,
        "status": "PASS",
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
