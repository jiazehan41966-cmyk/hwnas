import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec, StageSpec


SCRIPT = REPO_ROOT / "hls_lut_builder" / "scripts" / "generate_formal_lut.py"
CONFIG = REPO_ROOT / "configs" / "search" / "formal_lut_strict40_nksid_short_av7k325.yaml"


def _generate_strict40(tmpdir: Path) -> tuple[Path, Path]:
    lut_path = tmpdir / "formal_lut_strict40_v1.json"
    status_path = tmpdir / "formal_lut_status_strict40_v1.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--status-authoritative",
            "--output-lut-json",
            str(lut_path),
            "--output-status-json",
            str(status_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return lut_path, status_path


class Strict40LutSearchTests(unittest.TestCase):
    def test_status_authoritative_generation_produces_40_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path, status_path = _generate_strict40(Path(tmp))
            lut_payload = json.loads(lut_path.read_text(encoding="utf-8"))
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(len(lut_payload["entries"]), 40)
        self.assertTrue(lut_payload["metadata"]["status_authoritative"])
        self.assertEqual(lut_payload["metadata"]["measured_entries"], 40)
        self.assertEqual(lut_payload["metadata"]["board_result_status_overrides"], 0)
        self.assertEqual(
            status_payload["metadata"]["status_counts"],
            {"measured": 40, "defer_current_impl": 44},
        )

    def test_strict40_short_config_candidates_have_no_lut_misses_or_deferred_hits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path, status_path = _generate_strict40(Path(tmp))
            config = load_config(str(CONFIG))
            config["hardware"]["lut_path"] = str(lut_path)
            config["hardware"]["formal_lut_status_path"] = str(status_path)

            constraints = build_constraints(config)
            hardware_spec = build_hardware_spec(config)
            search_space = build_search_space(
                config,
                image_size=config["dataset"]["image_size"],
                input_channels=config["dataset"]["input_channels"],
                num_classes=config["dataset"]["num_classes"],
                constraints=constraints,
            )
            estimator = build_cost_estimator(
                config,
                hardware_spec=hardware_spec,
                constraints=constraints,
            )
            search_space, _ = apply_operator_policies_to_search_space(
                search_space,
                estimator.operator_policies,
            )

            effective_space = search_space.pre_prune(estimator)

            stage0_choice = effective_space.concrete_block_choices_for_stage(
                stage_index=0,
                in_channels=32,
                out_channels=16,
                stride=1,
            )[0]
            stage1_choices = effective_space.concrete_block_choices_for_stage(
                stage_index=1,
                in_channels=16,
                out_channels=24,
                stride=2,
            )
            stage2_choice = effective_space.concrete_block_choices_for_stage(
                stage_index=2,
                in_channels=24,
                out_channels=32,
                stride=2,
            )[0]
            stage3_choice = effective_space.concrete_block_choices_for_stage(
                stage_index=3,
                in_channels=32,
                out_channels=32,
                stride=1,
            )[0]
            self.assertEqual(len(stage1_choices), 4)
            self.assertIn(5, effective_space.config.kernel_choices)
            self.assertIn(6, effective_space.config.expand_choices)

            for choice in stage1_choices:
                architecture = ArchitectureSpec(
                    input_channels=1,
                    stem_channels=32,
                    stem_stride=2,
                    post_stem_downsample_stride=1,
                    head_conv_channels=None,
                    num_classes=8,
                    stages=(
                        StageSpec(16, 1, 1, (stage0_choice,)),
                        StageSpec(24, 1, 2, (choice,)),
                        StageSpec(32, 1, 2, (stage2_choice,)),
                        StageSpec(32, 1, 1, (stage3_choice,)),
                    ),
                )
                estimator.reset_lut_stats()
                cost = estimator.estimate(architecture, effective_space)
                stats = estimator.get_lut_stats()

                self.assertFalse(
                    any(violation.startswith("formal LUT ") for violation in cost.violations),
                    msg=f"{choice}: {cost.violations}",
                )
                self.assertEqual(stats["true_misses"], 0)
                self.assertEqual(stats["deferred_hits"], 0)
                self.assertEqual(stats["hits"], 5)


if __name__ == "__main__":
    unittest.main()
