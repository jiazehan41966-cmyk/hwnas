from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

from hwnas_fpga.models import build_model
from hwnas_fpga.search_space import ArchitectureSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_g5_candidate_jsons_build_models() -> None:
    expected_stage3 = {
        "mbconv_control": ("mbconv", "mbconv"),
        "denoise": ("denoise", "mbconv"),
        "edge": ("mbconv", "edge"),
        "denoise_edge": ("denoise", "edge"),
    }
    for variant, ops in expected_stage3.items():
        path = REPO_ROOT / "configs" / "ablation" / "sonar_g5_v1" / f"{variant}.candidate.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        architecture = ArchitectureSpec.from_dict(payload["candidate"]["encoding"])
        assert tuple(block.op for block in architecture.stages[3].blocks) == ops
        model = build_model(architecture, num_classes=8)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 1, 64, 64))
        assert tuple(output.shape) == (1, 8)


def test_bootstrap_uses_paired_predictions() -> None:
    script = _load_script("compare_sonar_ablation_bootstrap")
    control = [
        {"fold": 0, "seed": 42, "sample_index": 0, "target": 0, "prediction": 0},
        {"fold": 0, "seed": 42, "sample_index": 1, "target": 0, "prediction": 1},
        {"fold": 0, "seed": 42, "sample_index": 2, "target": 1, "prediction": 0},
        {"fold": 0, "seed": 42, "sample_index": 3, "target": 1, "prediction": 1},
    ]
    variant = [
        {"fold": 0, "seed": 42, "sample_index": 0, "target": 0, "prediction": 0},
        {"fold": 0, "seed": 42, "sample_index": 1, "target": 0, "prediction": 0},
        {"fold": 0, "seed": 42, "sample_index": 2, "target": 1, "prediction": 1},
        {"fold": 0, "seed": 42, "sample_index": 3, "target": 1, "prediction": 1},
    ]
    pairs = script.paired_rows(control, variant)
    result = script.stratified_bootstrap(
        pairs,
        num_classes=2,
        iterations=100,
        seed=7,
    )
    assert result["paired_prediction_count"] == 4
    assert result["macro_f1_mean_delta"] > 0.0


def test_make_synthetic_speckle_pairs_smoke(tmp_path: Path) -> None:
    script = _load_script("make_synthetic_speckle_pairs")
    data_root = tmp_path / "NKSID"
    for class_index, class_name in enumerate(("class_a", "class_b")):
        class_dir = data_root / class_name
        class_dir.mkdir(parents=True)
        for sample_index in range(2):
            Image.new("L", (8, 8), color=48 + class_index * 64 + sample_index).save(
                class_dir / f"img_{sample_index}.png"
            )
    with (data_root / "train_abs.txt").open("w", encoding="utf-8") as handle:
        for class_index, class_name in enumerate(("class_a", "class_b")):
            for sample_index in range(2):
                handle.write(f"{class_name}/img_{sample_index}.png {class_index}\n")

    output_dir = tmp_path / "synthetic"
    manifest = script.make_pairs(
        Namespace(
            data_dir=str(data_root),
            split="full",
            fold=0,
            image_size=8,
            levels="1",
            seed=20260707,
            max_samples=None,
            output_dir=str(output_dir),
            candidate_transforms="denoise",
        )
    )
    assert manifest["samples"] == 4
    assert (output_dir / "L1" / "ref").exists()
    assert (output_dir / "L1" / "noisy").exists()
    assert (output_dir / "L1" / "denoised").exists()


def test_build_sonar_operator_gate_manifest_merges_fragments(tmp_path: Path) -> None:
    script = _load_script("build_sonar_operator_gate_manifest")
    base = {
        "operators": {
            "denoise": {
                "quantization_contract": "per_tensor_symmetric_int8_v1",
                "software_spec_sha256": "TODO",
                "hls_spec_sha256": "TODO",
                "weight_export_complete": False,
                "weight_export_sha256": "TODO",
                "parity": {
                    "real_sample_count": 0,
                    "boundary_tensor_count": 0,
                    "random_tensor_count": 0,
                    "compared_element_count": 0,
                    "mismatch_count": 0,
                },
                "hls": {"evidence_complete": False, "route_feasible": False},
            },
            "edge": {
                "quantization_contract": "per_tensor_symmetric_int8_v1",
                "software_spec_sha256": "TODO",
                "hls_spec_sha256": "TODO",
                "weight_export_complete": False,
                "weight_export_sha256": "TODO",
                "parity": {
                    "real_sample_count": 0,
                    "boundary_tensor_count": 0,
                    "random_tensor_count": 0,
                    "compared_element_count": 0,
                    "mismatch_count": 0,
                },
                "hls": {"evidence_complete": False, "route_feasible": False},
            },
        },
        "ablation_variants": {},
        "comparisons_vs_control": {},
    }
    bootstrap = {
        "ablation_variants": {
            "mbconv_control": {
                "folds": [0, 1, 2, 3, 4],
                "seeds": [42, 43, 44],
                "completed_runs": 15,
                "claimable": True,
                "outer_leakage": False,
            }
        },
        "comparisons_vs_control": {
            "denoise": {
                "method": "paired_stratified_bootstrap",
                "iterations": 10000,
                "macro_f1_mean_delta": 0.01,
                "p_value": 0.01,
            }
        },
    }
    export = {
        "quantization_contract": "per_tensor_symmetric_int8_v1",
        "software_spec_sha256": "a" * 64,
        "hls_spec_sha256": "a" * 64,
        "weight_export_complete": True,
        "weight_export_sha256": "b" * 64,
        "manifest_path": str(tmp_path / "folded_export_manifest.json"),
    }
    base_path = tmp_path / "base.json"
    bootstrap_path = tmp_path / "bootstrap.json"
    export_path = tmp_path / "export.json"
    parity_path = tmp_path / "parity.jsonl"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    bootstrap_path.write_text(json.dumps(bootstrap), encoding="utf-8")
    export_path.write_text(json.dumps(export), encoding="utf-8")
    parity_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"input_kind": "real_sample", "element_count": 4, "mismatch_count": 0},
                {"input_kind": "boundary_tensor", "element_count": 4, "mismatch_count": 0},
                {"input_kind": "random_tensor", "element_count": 4, "mismatch_count": 0},
            )
        ),
        encoding="utf-8",
    )

    manifest = script.assemble_manifest(
        Namespace(
            base=str(base_path),
            bootstrap=str(bootstrap_path),
            denoise_export=str(export_path),
            edge_export=None,
            denoise_parity_records=str(parity_path),
            edge_parity_records=None,
            denoise_hls_evidence=None,
            edge_hls_evidence=None,
            output=str(tmp_path / "manifest.json"),
        )
    )
    assert manifest["ablation_variants"]["mbconv_control"]["completed_runs"] == 15
    assert manifest["operators"]["denoise"]["weight_export_complete"] is True
    assert manifest["operators"]["denoise"]["parity"]["real_sample_count"] == 1
    assert manifest["operators"]["denoise"]["parity"]["mismatch_count"] == 0
    assert "edge_export" not in manifest["source_fragments"]
