import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware.validation_report import build_final_hardware_validation_report


class HardwareValidationReportTests(unittest.TestCase):
    def test_claims_real_latency_only_with_route_clean_com5_measurement(self) -> None:
        candidate = {
            "candidate": {
                "arch_id": "arch_clean",
                "metrics": {
                    "latency_ms": 7.29,
                    "lut": 128445,
                    "dsp": 1172,
                    "bram": 188,
                    "power_w": 15.74,
                },
            }
        }
        route = {
            "status": "success",
            "wns_ns": "0.094",
            "timing_clean": True,
            "global_congestion_level": "3",
            "bitstream": r"E:\route\model.bit",
            "bitstream_sha256": "abc",
        }
        measurement = {
            "bitstream": r"E:\route\model.bit",
            "serial_port": "COM5",
            "frame": {"status_code": 0, "cycles": 9812402},
            "latency_ms": 49.06201,
            "clock_freq_hz": 200_000_000,
        }

        report = build_final_hardware_validation_report(
            candidate_payload=candidate,
            route_payload=route,
            measurement_payload=measurement,
            measurement_json_path=r"E:\route\measurement.json",
        )

        self.assertEqual(report["status"], "valid_board_latency")
        self.assertTrue(report["claimable_real_e2e_latency"])
        self.assertEqual(report["nas_lut_estimate"]["latency_ms"], 7.29)
        self.assertEqual(report["real_board_e2e"]["e2e_latency_ms"], 49.06201)
        self.assertEqual(report["real_board_e2e"]["e2e_cycles"], 9812402)
        self.assertIn("NAS strict LUT aggregate estimate", report["nas_lut_estimate"]["latency_scope"])

    def test_blocks_latency_when_route_is_not_clean_even_if_measurement_exists(self) -> None:
        report = build_final_hardware_validation_report(
            candidate_payload={"candidate": {"arch_id": "arch_fail", "metrics": {"latency_ms": 7.0}}},
            route_payload={
                "status": "failed",
                "wns_ns": "-3.484",
                "timing_clean": False,
                "global_congestion_level": "6",
                "bitstream": r"E:\route\model.bit",
            },
            measurement_payload={
                "bitstream": r"E:\route\model.bit",
                "serial_port": "COM5",
                "frame": {"status_code": 0, "cycles": 100},
                "latency_ms": 0.5,
            },
            measurement_json_path=r"E:\route\measurement.json",
        )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["claimable_real_e2e_latency"])
        self.assertIn("full_route_not_clean", report["blockers"])
        self.assertIsNone(report["real_board_e2e"]["e2e_latency_ms"])

    def test_blocks_latency_when_measurement_status_is_nonzero(self) -> None:
        report = build_final_hardware_validation_report(
            candidate_payload={"candidate": {"arch_id": "arch_fail", "metrics": {"latency_ms": 7.0}}},
            route_payload={
                "status": "success",
                "wns_ns": "0.1",
                "timing_clean": True,
                "bitstream": r"E:\route\model.bit",
            },
            measurement_payload={
                "bitstream": r"E:\route\model.bit",
                "serial_port": "COM5",
                "frame": {"status_code": 2, "cycles": 100},
                "latency_ms": 0.5,
            },
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("measurement_status_code_2", report["blockers"])
        self.assertEqual(report["real_board_e2e"]["raw_latency_ms"], 0.5)
        self.assertIsNone(report["real_board_e2e"]["e2e_latency_ms"])


if __name__ == "__main__":
    unittest.main()
