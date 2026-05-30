import hashlib
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

from hwnas_fpga.hardware.lookup_table import LutQueryEngine, LutTable, OpSpec
from hwnas_fpga.runtime import (
    apply_operator_policies_to_search_space,
    build_constraints,
    build_cost_estimator,
    build_hardware_spec,
    build_search_space,
    load_config,
)
from hwnas_fpga.search_space import ArchitectureSpec, BlockSpec, StageSpec


SCRIPT = (
    REPO_ROOT
    / "hls_lut_builder"
    / "scripts"
    / "generate_nas_board_lut_current84_arch84.py"
)
CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_smoke_av7k325.yaml"
)
SHORT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "search"
    / "nas_board_lut_strict_current84_arch84_nksid_short_av7k325.yaml"
)
PROTECTED_RAW_INPUTS = (
    REPO_ROOT
    / "hls_lut_builder"
    / "results"
    / "v2_deployable_lut"
    / "deployable_lut_v2.json",
    REPO_ROOT
    / "hls_lut_builder"
    / "results"
    / "v2_current84_deployable_lut"
    / "deployable_lut_v2_current84.json",
    REPO_ROOT
    / "hls_lut_builder"
    / "board_harness"
    / "results"
    / "board_measured_lut.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_to(output_dir: Path) -> tuple[Path, Path, dict]:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    summary_path = output_dir / "summary.json"
    return (
        output_dir / "nas_board_lut_strict.json",
        output_dir / "nas_board_lut_status.json",
        json.loads(summary_path.read_text(encoding="utf-8")),
    )


class NasBoardLutStrictCurrent84Arch84Tests(unittest.TestCase):
    def test_generator_counts_and_preserves_raw_ledgers(self) -> None:
        before = {path: _sha256(path) for path in PROTECTED_RAW_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            _lut_path, _status_path, summary = _generate_to(Path(tmp))
        after = {path: _sha256(path) for path in PROTECTED_RAW_INPUTS}

        self.assertEqual(before, after)
        e2e_measured = summary["by_source_status"].get(
            "sonar_classifier_e2e_official", {}
        ).get("measured", 0)
        self.assertIn(e2e_measured, (0, 1))
        self.assertEqual(summary["strict_usable_count"], 57 + e2e_measured)
        self.assertEqual(summary["timing_fail_reference_count"], 30)
        self.assertEqual(summary["missing_unusable_count"], 1 - e2e_measured)
        self.assertEqual(summary["by_source_status"]["current84_fullcombo84"]["measured"], 54)
        self.assertEqual(
            summary["by_source_status"]["current84_fullcombo84"][
                "timing_fail_reference"
            ],
            30,
        )
        self.assertEqual(summary["by_source_status"]["arch84_e2e_extension"]["measured"], 3)
        self.assertEqual(summary["unparsed_case_count"], 0)

    def test_lut_loader_keeps_timing_fail_reference_out_of_measured_lut(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path, status_path, _summary = _generate_to(Path(tmp))

            table = LutTable.load_json(str(lut_path))
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            e2e_measured = sum(
                1
                for entry in status_payload["entries"]
                if entry["source_namespace"] == "sonar_classifier_e2e_official"
                and entry["status"] == "measured"
            )
            self.assertEqual(len(table), 57 + e2e_measured)

            timing_fail_raw = next(
                entry
                for entry in status_payload["entries"]
                if entry["status"] == "timing_fail_reference"
            )
            timing_fail_spec = OpSpec.from_dict(timing_fail_raw["op_spec"])
            status_entries = LutQueryEngine.load_formal_status_json(str(status_path))
            engine = LutQueryEngine(
                table,
                enable_interpolation=False,
                allow_shape_only_match=False,
                strict_formal_lut=True,
                formal_status_entries=status_entries,
            )

            self.assertIsNone(table.query(timing_fail_spec))
            result = engine.query_with_status(timing_fail_spec)
            self.assertEqual(result.status, "timing_fail_reference")
            self.assertIsNone(result.entry)

    def test_smoke_config_evaluates_exact_arch84_candidate_without_strict_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path, status_path, _summary = _generate_to(Path(tmp))

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

            architecture = ArchitectureSpec(
                input_channels=1,
                stem_channels=32,
                stem_stride=2,
                post_stem_downsample_stride=4,
                head_conv_channels=None,
                num_classes=8,
                stages=(
                    StageSpec(
                        channels=32,
                        depth=1,
                        stride=1,
                        blocks=(BlockSpec("mbconv", kernel_size=3, expand_ratio=3),),
                    ),
                ),
            )
            estimator.reset_lut_stats()
            cost = estimator.estimate(architecture, search_space)
            stats = estimator.get_lut_stats()

            self.assertFalse(
                any(
                    violation.startswith("strict board LUT ")
                    for violation in cost.violations
                ),
                msg=cost.violations,
            )
            self.assertEqual(stats["strict_lut_label"], "strict board LUT")
            self.assertTrue(stats["use_lut_for_head_layers"])
            self.assertEqual(stats["true_misses"], 0)
            self.assertEqual(stats["deferred_hits"], 0)
            self.assertGreater(stats["hits"], 0)
            self.assertGreaterEqual(stats["hits"], 4)

    def test_short_nksid_config_all_choices_have_strict_lut_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lut_path, status_path, _summary = _generate_to(Path(tmp))

            config = load_config(str(SHORT_CONFIG))
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
            search_space, _policy_summary = apply_operator_policies_to_search_space(
                search_space,
                estimator.operator_policies,
            )

            stage0 = search_space.concrete_block_choices_for_stage(
                stage_index=0,
                in_channels=32,
                out_channels=16,
                stride=1,
            )
            stage1 = search_space.concrete_block_choices_for_stage(
                stage_index=1,
                in_channels=16,
                out_channels=24,
                stride=2,
            )
            stage2 = search_space.concrete_block_choices_for_stage(
                stage_index=2,
                in_channels=24,
                out_channels=32,
                stride=2,
            )
            stage3 = search_space.concrete_block_choices_for_stage(
                stage_index=3,
                in_channels=32,
                out_channels=32,
                stride=1,
            )

            self.assertIsNotNone(stage0)
            self.assertIsNotNone(stage1)
            self.assertIsNotNone(stage2)
            self.assertIsNotNone(stage3)
            self.assertEqual(len(stage1), 4)
            self.assertEqual(len(stage3), 2)

            for stage1_block in stage1:
                for stage3_block in stage3:
                    architecture = ArchitectureSpec(
                        input_channels=1,
                        stem_channels=32,
                        stem_stride=2,
                        post_stem_downsample_stride=1,
                        head_conv_channels=None,
                        num_classes=8,
                        stages=(
                            StageSpec(16, 1, 1, (stage0[0],)),
                            StageSpec(24, 1, 2, (stage1_block,)),
                            StageSpec(32, 1, 2, (stage2[0],)),
                            StageSpec(32, 1, 1, (stage3_block,)),
                        ),
                    )
                    estimator.reset_lut_stats()
                    cost = estimator.estimate(architecture, search_space)
                    stats = estimator.get_lut_stats()

                    self.assertEqual(cost.violations, ())
                    self.assertEqual(stats["true_misses"], 0)
                    self.assertEqual(stats["deferred_hits"], 0)
                    self.assertEqual(stats["hits"], 7)


if __name__ == "__main__":
    unittest.main()
