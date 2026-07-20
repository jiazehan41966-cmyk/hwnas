import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.hardware import FPGACostEstimator
from hwnas_fpga.interfaces import HardwareSpec
from hwnas_fpga.runtime import load_analytic_calibration
from hwnas_fpga.search_space import SearchSpace, SearchSpaceConfig


def _make_estimator(calibration=None) -> FPGACostEstimator:
    return FPGACostEstimator(
        hardware_spec=HardwareSpec(
            name="test-fpga",
            clock_mhz=200,
            max_lut=203_800,
            max_bram=445,
            max_dsp=840,
        ),
        analytic_calibration=calibration,
    )


class AnalyticCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = SearchSpace(SearchSpaceConfig(num_classes=8))
        self.architecture = self.space.baseline_architecture()

    def test_identity_calibration_matches_uncalibrated(self) -> None:
        baseline = _make_estimator().estimate(self.architecture, self.space)
        identity = _make_estimator(
            {"latency_scale": 1.0, "dsp_scale": 1.0, "lut_scale": 1.0, "bram_scale": 1.0}
        ).estimate(self.architecture, self.space)
        self.assertEqual(baseline.total_dsp, identity.total_dsp)
        self.assertEqual(baseline.total_lut, identity.total_lut)
        self.assertEqual(baseline.latency_cycles, identity.latency_cycles)

    def test_calibration_scales_analytic_estimates(self) -> None:
        baseline = _make_estimator().estimate(self.architecture, self.space)
        calibrated = _make_estimator(
            {"latency_scale": 4.0, "dsp_scale": 0.5, "lut_scale": 0.2, "bram_scale": 0.7}
        ).estimate(self.architecture, self.space)

        # Latency scales up ~4x (per-layer rounding allows small drift).
        self.assertGreater(
            calibrated.latency_cycles, 3.5 * baseline.latency_cycles
        )
        # Resources scale down toward routed reality.
        self.assertLess(calibrated.total_dsp, 0.6 * baseline.total_dsp)
        self.assertLess(calibrated.total_lut, 0.3 * baseline.total_lut)
        self.assertLess(calibrated.total_bram, 0.85 * baseline.total_bram)

    def test_invalid_factor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_estimator({"latency_scale": 0.0})

    def test_partial_factors_default_to_identity(self) -> None:
        baseline = _make_estimator().estimate(self.architecture, self.space)
        partial = _make_estimator({"latency_scale": 2.0}).estimate(
            self.architecture, self.space
        )
        self.assertEqual(partial.total_dsp, baseline.total_dsp)
        self.assertGreater(partial.latency_cycles, 1.5 * baseline.latency_cycles)


class LoadCalibrationConfigTests(unittest.TestCase):
    def test_absent_key_returns_none(self) -> None:
        self.assertIsNone(load_analytic_calibration({"hardware": {}}))

    def test_missing_file_raises(self) -> None:
        config = {"hardware": {"analytic_calibration_path": "does/not/exist.json"}}
        with self.assertRaises(FileNotFoundError):
            load_analytic_calibration(config)

    def test_loads_recommended_block(self) -> None:
        payload = {
            "recommended_analytic_calibration": {
                "latency_scale": 7.16,
                "dsp_scale": 0.56,
                "lut_scale": 0.16,
                "bram_scale": 0.73,
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            factors = load_analytic_calibration(
                {"hardware": {"analytic_calibration_path": str(path)}}
            )
        self.assertAlmostEqual(factors["latency_scale"], 7.16)
        self.assertAlmostEqual(factors["lut_scale"], 0.16)

    def test_loads_flat_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.json"
            path.write_text(json.dumps({"latency_scale": 2.0}), encoding="utf-8")
            factors = load_analytic_calibration(
                {"hardware": {"analytic_calibration_path": str(path)}}
            )
        self.assertAlmostEqual(factors["latency_scale"], 2.0)


if __name__ == "__main__":
    unittest.main()
