"""Frozen Phase 5A bindings. X*, search space, NAS, and SonarOp are read-only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "tools/sonar_analysis", ROOT / "tools/sonar_spec", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from hwnas_fpga.models.sonar_candidate import SPEC_PATH, build_sonar_candidate
from hwnas_fpga.models.sonar_supernet import architecture_id
from hwnas_fpga.runtime import load_config

NETWORKS = {
    "B1": "P2_P3_P4_P5_s1k3_s2k5_s4k5_s5k5_s6k5_s7k5",
    "C": "P3_P4_P5_s1k3_s2k5_s4k5_s5k3_s6k5_s7k3",
    "D": "P3_P4_P5_s1k3_s2k3_s4k3_s5k3_s6k3_s7k3",
}
SEEDS = (42, 43, 44)
CANDIDATE_SET = ROOT / "reports/phase2f_multiseed/network_candidate_set.json"
MULTISEED_CSV = ROOT / "reports/phase2f_multiseed/multiseed_results.csv"
CHECKPOINT_SELECTION_MANIFEST = ROOT / "reports/phase5a_quantization/CHECKPOINT_SELECTION_MANIFEST.json"
BASELINE_CONFIG = ROOT / "configs/baseline_sonar_frozen.yaml"
SEARCH_SPACE = ROOT / "configs/search_space_sonar_v0_1.yaml"
SEARCH_SPACE_SHA256 = "ebb1708b394af20258c920eb29204d5cc9504f5908fdfdae62a62d78042eb7c4"
REPORT = ROOT / "reports/phase5a_quantization"
ARTIFACTS = ROOT / "artifacts/phase5a_quantization"
QAT_ABSOLUTE_DROP = 0.003
QAT_RELATIVE_DROP = 0.20
H2_SUPPORTED_OPS = {
    "conv",
    "dw_conv",
    "pw_conv",
    "relu",
    "relu6",
    "linear",
    "nearest_upsample",
    "aligned_add",
    "identity_skip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def parse_architecture_id(value: str) -> dict[str, Any]:
    tokens = value.split("_")
    kernel_tokens = [token for token in tokens if token.startswith("s") and "k" in token]
    kernels = {}
    for token in kernel_tokens:
        stage, kernel = token[1:].split("k", 1)
        kernels[f"stage{int(stage)}"] = int(kernel)
    return {"output_combination": "_".join(tokens[: len(tokens) - len(kernel_tokens)]), "kernels": kernels}


def require_frozen_xstar() -> dict[str, str]:
    payload = json.loads(CANDIDATE_SET.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise ValueError("X* candidate set must remain test_evaluated=false")
    rows = {row["label"]: row["architecture_id"] for row in payload["architectures"]}
    if rows != NETWORKS:
        raise ValueError("Phase 5A cannot modify X* = {B1, C, D}")
    if sha256_file(SEARCH_SPACE) != SEARCH_SPACE_SHA256:
        raise ValueError("Search Space V0.1 hash changed")
    for label, identity in NETWORKS.items():
        architecture = parse_architecture_id(identity)
        if architecture_id(architecture, spec=SEARCH_SPACE) != identity:
            raise ValueError(f"{label} architecture id does not round-trip")
    return dict(NETWORKS)


def admit_network(label: str) -> str:
    require_frozen_xstar()
    if label not in NETWORKS:
        raise ValueError(f"Phase 5A admits only B1/C/D; rejected {label!r}")
    return NETWORKS[label]


def forbid_split(split: str) -> None:
    name = str(split).strip().lower()
    if name in {"test", "test1", "test2"}:
        raise ValueError("test split is sealed; test_evaluated=false")
    if name not in {"train", "valid"}:
        raise ValueError(f"unsupported split {split!r}")


def baseline() -> dict[str, Any]:
    return load_config(str(BASELINE_CONFIG))


def build_network(label: str):
    identity = admit_network(label)
    return build_sonar_candidate(SPEC_PATH, parse_architecture_id(identity)), identity


def checkpoint_rows() -> list[dict[str, str]]:
    manifest = json.loads(CHECKPOINT_SELECTION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY_FOR_PTQ":
        raise ValueError("checkpoint selection audit is not READY_FOR_PTQ")
    if manifest.get("frozen_checkpoint_policy") != "ALL_BEST_PT":
        raise ValueError("Phase5A PTQ requires frozen ALL_BEST_PT policy")
    rows = []
    for run in manifest["runs"]:
        label = str(run["network"])
        seed = int(run["seed"])
        if run.get("verification_status") != "VERIFIED_ORIGINAL":
            raise ValueError(f"{label} seed{seed} best.pt is not VERIFIED_ORIGINAL")
        if label not in NETWORKS:
            continue
        rows.append(
            {
                "label": label,
                "seed": str(seed),
                "architecture_id": str(run["architecture_id"]),
                "best_checkpoint_sha256": str(run["best_pt"]["sha256"]),
                "best_checkpoint": str(run.get("best_checkpoint_path") or ""),
                "mAP50_95": str(run["phase2f_reported_mAP50_95"]),
                "phase2f_bound_artifact": str(run.get("phase2f_bound_artifact", "")),
                "phase2f_metric_source": str(run.get("phase2f_metric_source", "")),
            }
        )
    rows.sort(key=lambda item: (list(NETWORKS).index(item["label"]), int(item["seed"])))
    if { (row["label"], int(row["seed"])) for row in rows } != {
        (label, seed) for label in NETWORKS for seed in SEEDS
    }:
        raise ValueError("checkpoint selection manifest is missing a frozen B1/C/D seed")
    return rows


def candidate_checkpoint_paths(label: str, seed: int, row: Mapping[str, str]) -> list[Path]:
    identity = admit_network(label)
    recorded = str(row.get("best_checkpoint") or "").strip()
    names = []
    if recorded:
        names.append(Path(recorded))
    names.extend(
        [
            ROOT / "results/phase2f_multiseed" / identity / f"seed_{seed}" / "best.pt",
            ROOT / "results/phase2e_independent" / identity / "best.pt" if seed == 42 else None,
            ARTIFACTS / "checkpoints" / identity / f"seed_{seed}" / "best.pt",
        ]
    )
    root = Path(__import__("os").environ.get("PHASE5A_CHECKPOINT_ROOT", "")).expanduser()
    if str(root):
        names.append(root / identity / f"seed_{seed}" / "best.pt")
        names.append(root / "results/phase2f_multiseed" / identity / f"seed_{seed}" / "best.pt")
        if seed == 42:
            names.append(root / "results/phase2e_independent" / identity / "best.pt")
    return [path for path in names if path is not None]


def resolve_checkpoint(label: str, seed: int, row: Mapping[str, str]) -> dict[str, Any]:
    expected = str(row["best_checkpoint_sha256"])
    for path in candidate_checkpoint_paths(label, seed, row):
        if path.is_file():
            digest = sha256_file(path)
            if digest != expected:
                raise ValueError(f"checkpoint hash mismatch for {label} seed{seed}: {path}")
            return {
                "path": str(path),
                "sha256": digest,
                "status": "FOUND",
                "label": label,
                "seed": int(seed),
                "architecture_id": row["architecture_id"],
            }
    return {
        "path": None,
        "sha256": expected,
        "status": "WAITING_FOR_CHECKPOINT",
        "label": label,
        "seed": int(seed),
        "architecture_id": row["architecture_id"],
    }
