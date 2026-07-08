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
