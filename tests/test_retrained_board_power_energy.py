import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "hls_lut_builder"
    / "board_harness"
    / "scripts"
    / "import_retrained_board_power_energy.py"
)

spec = importlib.util.spec_from_file_location("import_retrained_board_power_energy", SCRIPT_PATH)
power_importer = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = power_importer
spec.loader.exec_module(power_importer)


class RetrainedBoardPowerEnergyTests(unittest.TestCase):
    def _write_csv(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _fake_root(self, root: Path) -> None:
        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "retrained_board_reinject_comparison.json").write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "run_name": "rl_arch_186_retrain150_trainedmem",
                            "arch_id": "rl_arch_186",
                            "positioning": "accuracy-first",
                            "bitstream_sha256": "sha186",
                            "retrain_macro_f1": 0.85,
                            "retrain_top1": 0.91,
                            "real_e2e_latency_ms": 10.0,
                        },
                        {
                            "run_name": "rl_arch_242_retrain150_trainedmem",
                            "arch_id": "rl_arch_242",
                            "positioning": "latency-efficient",
                            "bitstream_sha256": "sha242",
                            "retrain_macro_f1": 0.84,
                            "retrain_top1": 0.91,
                            "real_e2e_latency_ms": 5.0,
                        },
                    ]
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_power_csv_and_voltage_current_csv_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            power_csv = tmp_path / "power.csv"
            vi_csv = tmp_path / "vi.csv"
            self._write_csv(power_csv, "timestamp_s,power_w\n0,5\n1,7\n2,9\n")
            self._write_csv(vi_csv, "timestamp_s,voltage_v,current_a\n0,5,1\n1,5,2\n")

            power = power_importer.load_power_csv(power_csv)
            vi = power_importer.load_power_csv(vi_csv)

        self.assertTrue(power["timestamp_monotonic"])
        self.assertEqual(power["sample_count"], 3)
        self.assertAlmostEqual(power["energy_j"], 14.0)
        self.assertAlmostEqual(power["time_weighted_mean_w"], 7.0)
        self.assertAlmostEqual(vi["energy_j"], 7.5)

    def test_import_computes_energy_and_outputs_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_root(root)
            manifest = root / "power_measurements" / "power_measurement_manifest.json"
            self._write_csv(root / "power_measurements" / "raw" / "idle.csv", "timestamp_s,power_w\n0,5\n1,5\n2,5\n")
            self._write_csv(root / "power_measurements" / "raw" / "active.csv", "timestamp_s,power_w\n0,7\n1,7\n2,7\n")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "measurement_source": "csv_import",
                        "rail_scope": "board_input_total",
                        "instrument": {"model": "unit", "sample_rate_hz": 1, "calibration": "synthetic"},
                        "measurements": [
                            {
                                "arch_id": "rl_arch_186",
                                "run_name": "rl_arch_186_retrain150_trainedmem",
                                "bitstream_sha256": "sha186",
                                "idle_csv": "raw/idle.csv",
                                "active_csvs": ["raw/active.csv"],
                                "active_inference_count": 2,
                                "contains_programming": False,
                                "same_com5_input": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = power_importer.import_power_energy(root, manifest)
            summary = json.loads(
                (root / "power_measurements" / "reports" / "retrained_board_power_energy_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            audit = json.loads(
                (
                    root
                    / "power_measurements"
                    / "reports"
                    / "retrained_board_power_energy_acceptance_audit.json"
                ).read_text(encoding="utf-8")
            )

        row = summary["rows"][0]
        self.assertTrue(result["overall_pass"])
        self.assertTrue(audit["overall_pass"])
        self.assertAlmostEqual(row["idle_power_mean_w"], 5.0)
        self.assertAlmostEqual(row["active_power_mean_w"], 7.0)
        self.assertAlmostEqual(row["dynamic_power_mean_w"], 2.0)
        self.assertAlmostEqual(row["total_energy_mj"], 7000.0)
        self.assertAlmostEqual(row["dynamic_energy_mj"], 2000.0)
        self.assertAlmostEqual(row["latency_dynamic_energy_mj"], 20.0)

    def test_multiple_active_csvs_aggregate_mean_and_std(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_root(root)
            manifest = root / "power_measurements" / "power_measurement_manifest.json"
            self._write_csv(root / "power_measurements" / "raw" / "idle.csv", "timestamp_s,power_w\n0,5\n2,5\n")
            self._write_csv(root / "power_measurements" / "raw" / "active1.csv", "timestamp_s,power_w\n0,7\n2,7\n")
            self._write_csv(root / "power_measurements" / "raw" / "active2.csv", "timestamp_s,power_w\n0,9\n2,9\n")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "measurement_source": "csv_import",
                        "rail_scope": "board_input_total",
                        "instrument": {"model": "unit"},
                        "measurements": [
                            {
                                "arch_id": "rl_arch_186",
                                "run_name": "rl_arch_186_retrain150_trainedmem",
                                "bitstream_sha256": "sha186",
                                "idle_csv": "raw/idle.csv",
                                "active_csvs": ["raw/active1.csv", "raw/active2.csv"],
                                "active_inference_count": 4,
                                "contains_programming": False,
                                "same_com5_input": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            power_importer.import_power_energy(root, manifest)
            summary = json.loads(
                (root / "power_measurements" / "reports" / "retrained_board_power_energy_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        row = summary["rows"][0]
        self.assertAlmostEqual(row["active_power_mean_w"], 8.0)
        self.assertAlmostEqual(row["active_power_mean_std_w"], 1.0)
        self.assertAlmostEqual(row["dynamic_energy_mj"], 3000.0)

    def test_audit_fails_nonmonotonic_timestamp_and_bad_manifest_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_root(root)
            manifest = root / "power_measurements" / "power_measurement_manifest.json"
            self._write_csv(root / "power_measurements" / "raw" / "idle.csv", "timestamp_s,power_w\n0,5\n1,5\n")
            self._write_csv(root / "power_measurements" / "raw" / "bad_active.csv", "timestamp_s,power_w\n0,7\n0,7\n")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "measurement_source": "csv_import",
                        "rail_scope": "board_input_total",
                        "instrument": {"model": "unit"},
                        "measurements": [
                            {
                                "arch_id": "rl_arch_186",
                                "run_name": "rl_arch_186_retrain150_trainedmem",
                                "bitstream_sha256": "wrong",
                                "idle_csv": "raw/idle.csv",
                                "active_csvs": ["raw/bad_active.csv"],
                                "active_inference_count": 1,
                                "contains_programming": True,
                                "same_com5_input": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = power_importer.import_power_energy(root, manifest)
            audit = json.loads(
                (
                    root
                    / "power_measurements"
                    / "reports"
                    / "retrained_board_power_energy_acceptance_audit.json"
                ).read_text(encoding="utf-8")
            )

        gates = {gate["gate"]: gate["pass"] for gate in audit["runs"][0]["gates"]}
        self.assertFalse(result["overall_pass"])
        self.assertFalse(gates["active_timestamp_monotonic"])
        self.assertFalse(gates["bitstream_sha256_matches_report"])
        self.assertFalse(gates["contains_programming_false"])
        self.assertFalse(gates["same_com5_input_true"])

    def test_audit_fails_negative_dynamic_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fake_root(root)
            manifest = root / "power_measurements" / "power_measurement_manifest.json"
            self._write_csv(root / "power_measurements" / "raw" / "idle.csv", "timestamp_s,power_w\n0,9\n1,9\n")
            self._write_csv(root / "power_measurements" / "raw" / "active.csv", "timestamp_s,power_w\n0,7\n1,7\n")
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "measurement_source": "csv_import",
                        "rail_scope": "board_input_total",
                        "instrument": {"model": "unit"},
                        "measurements": [
                            {
                                "arch_id": "rl_arch_186",
                                "run_name": "rl_arch_186_retrain150_trainedmem",
                                "bitstream_sha256": "sha186",
                                "idle_csv": "raw/idle.csv",
                                "active_csvs": ["raw/active.csv"],
                                "active_inference_count": 1,
                                "contains_programming": False,
                                "same_com5_input": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = power_importer.import_power_energy(root, manifest)
            audit = json.loads(
                (
                    root
                    / "power_measurements"
                    / "reports"
                    / "retrained_board_power_energy_acceptance_audit.json"
                ).read_text(encoding="utf-8")
            )

        gates = {gate["gate"]: gate["pass"] for gate in audit["runs"][0]["gates"]}
        self.assertFalse(result["overall_pass"])
        self.assertFalse(gates["dynamic_power_nonnegative"])


if __name__ == "__main__":
    unittest.main()
