import csv
import json
import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.hardware.power_measurement import load_and_audit_power_manifest


def _write_capture(path: Path, power: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_s", "power_w"])
        writer.writeheader()
        for second in range(61):
            writer.writerow({"timestamp_s": second, "power_w": power})


class PowerAuditTests(unittest.TestCase):
    def test_three_by_three_protocol_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            idle = []
            active = []
            for index in range(3):
                idle_path = root / f"idle_{index}.csv"
                active_path = root / f"active_{index}.csv"
                _write_capture(idle_path, 5.0)
                _write_capture(active_path, 7.0)
                idle.append(idle_path.name)
                receipt_path = root / f"receipt_{index}.json"
                receipt_path.write_text(
                    json.dumps(
                        {
                            "repeat_count": 1000,
                            "bitstream_sha256": "a" * 64,
                            "host_active_elapsed_s": 60.0,
                            "contains_programming": False,
                            "contains_uart_upload": False,
                        }
                    ),
                    encoding="utf-8",
                )
                active.append(
                    {
                        "csv": active_path.name,
                        "inference_count": 1000,
                        "run_repeat_receipt": receipt_path.name,
                    }
                )
            manifest = {
                "measurement_source": "external_power_meter_csv",
                "rail_scope": "board_input_total",
                "instrument": {
                    "model": "test",
                    "sample_rate_hz": 1,
                    "calibration": "traceable-test",
                },
                "bitstream_sha256": "a" * 64,
                "contains_programming": False,
                "contains_uart_upload": False,
                "idle_csvs": idle,
                "active_captures": active,
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = load_and_audit_power_manifest(path)
            self.assertTrue(result["overall_pass"])
            self.assertAlmostEqual(result["dynamic_power_mean_w"], 2.0)
            self.assertAlmostEqual(
                result["dynamic_energy_mj_per_inference"],
                120.0,
            )

    def test_short_or_single_capture_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            capture = root / "capture.csv"
            _write_capture(capture, 5.0)
            path = root / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "measurement_source": "external_power_meter_csv",
                        "rail_scope": "board_input_total",
                        "instrument": {
                            "model": "test",
                            "sample_rate_hz": 1,
                            "calibration": "traceable-test",
                        },
                        "bitstream_sha256": "a" * 64,
                        "contains_programming": False,
                        "contains_uart_upload": False,
                        "idle_csvs": [capture.name],
                        "active_captures": [
                            {"csv": capture.name, "inference_count": 10}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            result = load_and_audit_power_manifest(path)
            self.assertFalse(result["overall_pass"])


if __name__ == "__main__":
    unittest.main()
