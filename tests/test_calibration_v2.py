import unittest

from hwnas_fpga.hardware.calibration_v2 import (
    architecture_family,
    classify_intervals,
    deduplicate_pairs,
    evidence_fingerprint,
    fit_ratio_model,
    fit_robust_affine,
    validate_affine_independent,
    validate_ratio_model,
    validate_ratio_model_independent,
    validation_gate,
)


def _row(index: int, *, estimated: float, measured: float) -> dict:
    return {
        "fingerprint": f"fp-{index}",
        "family": "mainline_mbconv_skip",
        "estimated": {"lut": estimated},
        "measured": {"lut": measured},
    }


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_is_canonical(self) -> None:
        self.assertEqual(
            evidence_fingerprint({"a": 1, "b": 2}),
            evidence_fingerprint({"b": 2, "a": 1}),
        )

    def test_sonar_family_is_separate(self) -> None:
        self.assertEqual(
            architecture_family({"stages": [{"blocks": [{"op": "edge"}]}]}),
            "semantic_mismatch:edge",
        )


class DeduplicationTests(unittest.TestCase):
    def test_repeated_architecture_counts_once(self) -> None:
        row = {
            "fingerprint": "same",
            "estimated": {"lut": 100},
            "measured": {"lut": 20},
            "run": "a",
        }
        result = deduplicate_pairs([row, {**row, "run": "b"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["repeat_count"], 2)

    def test_conflicting_duplicate_fails(self) -> None:
        left = {"fingerprint": "same", "estimated": {"lut": 100}, "measured": {"lut": 20}}
        right = {"fingerprint": "same", "estimated": {"lut": 100}, "measured": {"lut": 30}}
        with self.assertRaises(ValueError):
            deduplicate_pairs([left, right])


class ValidationTests(unittest.TestCase):
    def test_leave_one_out_reports_low_error_for_stable_ratio(self) -> None:
        rows = [_row(i, estimated=100 + i, measured=20 + 0.2 * i) for i in range(6)]
        result = validate_ratio_model(rows, metric="lut", budget=25)
        self.assertLess(result["p90_ape"], 0.01)
        self.assertEqual(result["false_rejects"], 0)
        self.assertTrue(validation_gate("lut", result)["hard_screening_enabled"])

    def test_latency_requires_ranking(self) -> None:
        rows = [
            {
                "fingerprint": f"fp-{i}",
                "family": "mainline_mbconv_skip",
                "estimated": {"latency_ms": float(i + 1)},
                "measured": {"latency_ms": float(6 - i)},
            }
            for i in range(5)
        ]
        result = validate_ratio_model(rows, metric="latency_ms", budget=10)
        self.assertFalse(validation_gate("latency_ms", result)["hard_screening_enabled"])

    def test_independent_ratio_validation_is_not_refit(self) -> None:
        train = [
            {
                "fingerprint": f"train-{index}",
                "family": "mbconv",
                "estimated": {"lut": value},
                "measured": {"lut": value * 2},
            }
            for index, value in enumerate((10, 20, 30, 40))
        ]
        probes = [
            {
                "fingerprint": f"probe-{index}",
                "family": "mbconv",
                "estimated": {"lut": value},
                "measured": {"lut": value * 2},
            }
            for index, value in enumerate((15, 25, 35, 45))
        ]
        model = fit_ratio_model(train, metric="lut")
        result = validate_ratio_model_independent(
            model,
            probes,
            metric="lut",
            budget=100,
        )
        self.assertEqual(result["validation_scope"], "independent_frozen_probe")
        self.assertEqual(result["mape"], 0.0)

    def test_affine_board_overhead_is_explicit(self) -> None:
        train = [
            {
                "estimated": {"cycles": value},
                "measured": {"cycles": 100 + 2 * value},
            }
            for value in (10, 20, 30, 40)
        ]
        probes = [
            {
                "fingerprint": f"p{value}",
                "estimated": {"cycles": value},
                "measured": {"cycles": 100 + 2 * value},
            }
            for value in (15, 25, 35, 45)
        ]
        model = fit_robust_affine(train)
        self.assertAlmostEqual(model["intercept"], 100)
        self.assertAlmostEqual(model["slope"], 2)
        validation = validate_affine_independent(model, probes)
        self.assertEqual(validation["mape"], 0.0)


class ScreeningTests(unittest.TestCase):
    def test_rejects_only_on_validated_lower_bound(self) -> None:
        intervals = {
            "lut": {"lower": 110, "point": 120, "upper": 130},
            "latency_ms": {"lower": 60, "point": 70, "upper": 80},
        }
        gates = {
            "lut": {"hard_screening_enabled": True},
            "latency_ms": {"hard_screening_enabled": False},
        }
        result = classify_intervals(
            intervals,
            {"lut": 100, "latency_ms": 50},
            gates,
        )
        self.assertEqual(result["status"], "certified_reject")
        self.assertEqual(result["reject_metrics"], ["lut"])
        self.assertIn("latency_ms", result["uncertain_metrics"])

    def test_unvalidated_metric_passes_to_hls(self) -> None:
        result = classify_intervals(
            {"lut": {"lower": 200, "point": 220, "upper": 250}},
            {"lut": 100},
            {"lut": {"hard_screening_enabled": False}},
        )
        self.assertEqual(result["status"], "uncertain")


if __name__ == "__main__":
    unittest.main()
