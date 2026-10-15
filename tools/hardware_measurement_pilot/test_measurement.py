"""Acceptance failures, not implementations mirrored as success-only tests."""
from copy import deepcopy
import math
import unittest

from tools.hardware_measurement_pilot.measurement import (
    REQUIRED_BUILD_STAGES, extrapolate, statistics_ns, summarize,
)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.spec = dict(case_id="dw3_int8_low", case_sha256="1"*64,
                         numeric_contract_sha256="2"*64, bitstream_sha256="3"*64,
                         expected_output_sha256="4"*64, clock_hz=200000000,
                         board_id="ZCU104-test-fixture", expected_elements=24)
        self.manifest = dict(cases=[self.spec], batches=[], phase_clock_evidence_verified=True)
        self.rows = []
        for batch in range(3):
            bid = f"batch{batch}"
            self.manifest["batches"].append(dict(case_id=self.spec["case_id"], batch_id=bid,
                independence_event="reload_bitstream", independence_id=f"load{batch}",
                clock_hz=200000000, clock_hz_end=200000000, cpu_governor="performance",
                temperature_c=40., temperature_c_end=41.,
                started_utc=f"2026-10-12T10:0{batch}:00+08:00", finished_utc=f"2026-10-12T10:0{batch}:50+08:00"))
            for phase, count in (("warmup", 50), ("measured", 1000)):
                for iteration in range(count):
                    row = {key: value for key, value in self.spec.items() if key != "expected_elements"}
                    row.update(phase=phase, iteration=iteration, compute_ns=1000, transfer_ns=2000,
                               call_ns=4000, compared_elements=24, mismatched_elements=0,
                               observed_output_sha256="4"*64, batch_id=bid, group_id=str(iteration),
                               group_size=1, group_call_total_ns=4000)
                    self.rows.append(row)

    def test_complete_raw_rows_pass_with_warmup_excluded(self):
        self.rows[0]["compute_ns"] = 1700
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["cases"][0]["scopes"]["compute"]["n"], 3000)
        self.assertEqual(result["cases"][0]["scopes"]["compute"]["mean_ns"], 1000)

    def test_insufficient_count(self):
        self.assertEqual(summarize(self.rows[:-1], self.manifest)["status"], "FAIL")

    def test_warmup_requirement_cannot_be_lowered_by_manifest(self):
        self.manifest["warmup_minimum"] = 1
        with self.assertRaisesRegex(ValueError, "warmup_minimum"):
            summarize(self.rows, self.manifest)

    def test_runner_environment_deviation_is_preserved(self):
        self.manifest["batches"][0]["condition_deviations"] = ["CPU_GOVERNOR_CHANGED_WITHIN_BATCH"]
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["protocol_conditions_met"])
        self.assertFalse(result["hardware_measurement_eligible"])
        self.assertEqual(result["condition_deviations"][0]["observed"], "CPU_GOVERNOR_CHANGED_WITHIN_BATCH")

    def test_nan_is_failed_never_trimmed(self):
        self.rows[57]["compute_ns"] = math.nan
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("exclusions forbidden" in error for error in result["cases"][0]["errors"]))

    def test_bad_hash_clock_parity_each_rejected(self):
        for field, value in (("bitstream_sha256", "5"*64), ("case_sha256", "5"*64),
                             ("clock_hz", 100000000), ("mismatched_elements", 1),
                             ("observed_output_sha256", "5"*64), ("compared_elements", 23)):
            rows = deepcopy(self.rows)
            rows[58][field] = value
            with self.subTest(field=field):
                self.assertEqual(summarize(rows, self.manifest)["status"], "FAIL")

    def test_cv_above_five_percent_rejected_for_pl_scopes(self):
        for scope in ("compute", "transfer"):
            rows = deepcopy(self.rows)
            for row in rows:
                if row["phase"] == "measured" and row["iteration"] % 2:
                    row[scope+"_ns"] *= 1.2
            result = summarize(rows, self.manifest)
            with self.subTest(scope=scope):
                self.assertEqual(result["status"], "FAIL")
                self.assertTrue(any(scope+": CV exceeds" in error for error in result["cases"][0]["errors"]))

    def test_warmup_after_measurement_rejected(self):
        rows = self.rows[50:1050] + self.rows[:50] + self.rows[1050:]
        self.assertEqual(summarize(rows, self.manifest)["status"], "FAIL")

    def test_at_least_three_independent_batches_required(self):
        self.manifest["batches"] = self.manifest["batches"][:2]
        self.assertEqual(summarize(self.rows[:2100], self.manifest)["status"], "FAIL")

    def test_host_cv_is_not_default_acceptance_gate(self):
        for row in self.rows:
            if row["phase"] == "measured" and row["iteration"] % 2:
                row["call_ns"] *= 2
                row["group_call_total_ns"] *= 2
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertGreater(result["cases"][0]["scopes"]["call"]["cv"], .05)
        self.assertIsNone(result["cases"][0]["host_median_stability"]["threshold"])

    def test_grouped_host_observations_not_pseudoreplicated(self):
        for row in self.rows:
            if row["phase"] == "measured":
                group = row["iteration"] // 16
                count = 16 if group < 62 else 8
                row.update(group_id=str(group), group_size=count, group_call_total_ns=4000*count)
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "PASS")
        case = result["cases"][0]
        self.assertEqual(case["scopes"]["compute"]["n"], 3000)
        self.assertEqual(case["scopes"]["call"]["n"], 189)
        self.assertEqual(case["host_group_size_distribution"], {8: 3, 16: 186})
        self.assertEqual(case["host_scopes_by_group_size"][16]["n"], 186)
        self.assertEqual(case["host_scopes_by_group_size"][8]["n"], 3)
        self.assertEqual(case["host_median_stability"]["comparability"], "MIXED_GROUP_SIZES_USE_N_STRATIFIED_STATISTICS")
        self.rows[-1]["group_call_total_ns"] += 1
        self.assertEqual(summarize(self.rows, self.manifest)["status"], "FAIL")

    def test_host_median_threshold_only_when_explicit_draft(self):
        for row in self.rows[2100:]:
            row["call_ns"] *= 2
            row["group_call_total_ns"] *= 2
        self.assertEqual(summarize(self.rows, self.manifest)["status"], "PASS")
        self.manifest.update(host_median_relative_limit=.05, host_limit_status="DRAFT")
        self.assertEqual(summarize(self.rows, self.manifest)["status"], "FAIL")

    def test_missing_optional_governor_temperature_preserves_data_with_deviation(self):
        self.manifest["batches"][0].update(cpu_governor="UNAVAILABLE", temperature_c=None)
        result = summarize(self.rows, self.manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["protocol_conditions_met"])
        self.assertFalse(result["hardware_measurement_eligible"])
        self.assertEqual(len(result["condition_deviations"]), 2)

    def test_overlapping_batches_are_not_independent(self):
        self.manifest["batches"][1]["started_utc"] = "2026-10-12T10:00:30+08:00"
        self.assertEqual(summarize(self.rows, self.manifest)["status"], "FAIL")

    def test_linear_p95_and_sample_sd(self):
        result = statistics_ns([1, 2, 3, 4])
        self.assertAlmostEqual(result["p95_ns"], 3.85)
        self.assertAlmostEqual(result["sample_std_ns"], math.sqrt(5/3))

    def records(self):
        return [dict(case_id=f"case{i}", attempt_id="a1", stratum="DW3", status="COMPLETE",
                     parity_pass=True, timing_pass=True,
                     stages={stage: 10 for stage in REQUIRED_BUILD_STAGES}) for i in range(2)]

    def test_no_estimate_without_all_stages_or_family_coverage(self):
        records = self.records()
        records[0]["stages"].pop("route")
        result = extrapolate(records, {"DW3": 10}, physical_bindings_complete=True)
        self.assertIsNone(result["sequential_days"])
        result = extrapolate(self.records(), {"DW3": 10, "maxpool": 3}, physical_bindings_complete=True)
        self.assertIsNone(result["sequential_days"])
        self.assertIn("maxpool", result["missing_strata"])

    def test_nominal_policy_count_cannot_be_physical_estimate(self):
        result = extrapolate(self.records(), {"DW3": 10})
        self.assertIsNone(result["sequential_days"])

    def test_failed_attempt_duration_is_charged(self):
        records = self.records()
        records.append(dict(case_id="case2", attempt_id="a1", stratum="DW3", status="FAILED",
                            stages={"hls_synthesis": 60}))
        result = extrapolate(records, {"DW3": 10}, physical_bindings_complete=True)
        self.assertAlmostEqual(result["strata"][0]["seconds_per_success_including_failures"], 90)
        self.assertAlmostEqual(result["sequential_days"], 900/86400)


if __name__ == "__main__":
    unittest.main()
