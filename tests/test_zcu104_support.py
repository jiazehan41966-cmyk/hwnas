import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hls_lut_builder" / "board_harness" / "scripts"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_harness import (  # noqa: E402
    build_clock_port_declarations,
    build_clocking_logic,
)
from hwnas_fpga.hardware import (  # noqa: E402
    audit_zcu104_deployment_binding,
    get_board_profile,
)
from hwnas_fpga.hardware.power_measurement import load_and_audit_power_manifest  # noqa: E402
from hwnas_fpga.runtime import build_hardware_spec  # noqa: E402
from run_fourstage_external_scratch_hls_gate import (  # noqa: E402
    ZCU104_RESOURCE_LIMITS,
    ZCU104_TARGET_PART,
    check_hls_resources,
    tcl_script,
)
from generate_zcu104_platform import build_tcl as build_zcu104_platform_tcl  # noqa: E402


class Zcu104BoardSupportTests(unittest.TestCase):
    def test_board_profile_uses_explicit_bram_units(self) -> None:
        profile = get_board_profile("xczu7ev-ffvc1156-2-e")

        self.assertEqual(profile.name, "AMD/Xilinx ZCU104 (XCZU7EV-2FFVC1156)")
        self.assertEqual(profile.max_lut, 230_400)
        self.assertEqual(profile.max_dsp, 1_728)
        self.assertEqual(profile.max_bram, 312)
        self.assertEqual(profile.max_bram18, 624)
        self.assertEqual(profile.max_bram36, 312)
        self.assertEqual(profile.max_uram, 96)

        overridden = build_hardware_spec(
            {"hardware": {"board": "zcu104", "max_bram18": 600, "max_uram": 90}}
        )
        self.assertEqual(overridden.max_bram18, 600)
        self.assertEqual(overridden.max_uram, 90)

    def test_board_config_uses_ps_fclk_and_axi_dma(self) -> None:
        config = yaml.safe_load(
            (ROOT / "hls_lut_builder" / "board_harness" / "configs" / "board_zcu104.yaml")
            .read_text(encoding="utf-8")
        )

        self.assertEqual(config["target_part"], ZCU104_TARGET_PART)
        self.assertEqual(config["clock"]["source"], "ps_fclk")
        self.assertEqual(config["transport"]["type"], "axi_dma")
        self.assertTrue(config["platform_exports"]["xsa"])
        self.assertEqual(config["power"]["pmbus_role"], "auxiliary_only")

    def test_campaign_config_binds_the_revision_source_freeze(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "experiments" / "zcu104_full_campaign_v1.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["protocol"]["source_freeze_manifest"],
            "artifacts/zcu104_full_campaign_v1/"
            "source_freeze_revision_20260809/source_freeze_manifest.json",
        )

    def test_clock_generator_bypasses_7_series_mmcm_for_ps_fclk(self) -> None:
        config = yaml.safe_load(
            (ROOT / "hls_lut_builder" / "board_harness" / "configs" / "board_zcu104.yaml")
            .read_text(encoding="utf-8")
        )

        ports = build_clock_port_declarations(config)
        logic = build_clocking_logic(config)
        self.assertIn("pl_clk0", ports)
        self.assertIn("assign clk_main = pl_clk0", logic)
        self.assertNotIn("MMCME2_BASE", logic)

    def test_hls_tcl_and_resource_gate_are_zcu104_specific(self) -> None:
        tcl = tcl_script(target_part=ZCU104_TARGET_PART, target_clock_ns=5.0)
        self.assertIn(f"set_part {{{ZCU104_TARGET_PART}}}", tcl)
        check = check_hls_resources(
            {"lut": 112_137, "ff": 109_753, "dsp": 274, "bram_18k": 30},
            ZCU104_RESOURCE_LIMITS,
        )
        self.assertEqual(check["status"], "PASS")
        self.assertEqual(check["accounting"]["bram"], "BRAM18")

    def test_platform_tcl_connects_dma_scratch_ddr_interrupts_and_exports(self) -> None:
        tcl = build_zcu104_platform_tcl(
            ip_repo=Path("/tmp/hls_ip"),
            output_dir=Path("/tmp/zcu104_platform"),
            kernel_vlnv="xilinx.com:hls:fourstage_top_hls:1.0",
            project_name="zcu104_fourstage",
            jobs=8,
        )

        self.assertIn("xilinx.com:ip:axi_dma:7.1", tcl)
        self.assertIn("kernel/input_stream", tcl)
        self.assertIn("kernel/m_axi_gmem3", tcl)
        self.assertIn("ps/S_AXI_HPC0_FPD", tcl)
        self.assertIn("ps/pl_ps_irq0", tcl)
        self.assertIn("write_hw_platform -fixed -include_bit", tcl)
        self.assertIn("zcu104_fourstage.hwh", tcl)

    def test_deployment_binding_requires_all_sha_bound_artifacts(self) -> None:
        record = {"path": "/tmp/item", "sha256": "a" * 64}
        payload = {
            "campaign_id": "zcu104_full_campaign_v1",
            "candidate": {"arch_id": "arch", "architecture_sha256": "b" * 64},
            "platform": {
                "board_model": "ZCU104",
                "target_part": ZCU104_TARGET_PART,
                "board_serial": "36708",
            },
            "transport": {"type": "axi_dma", "id": "axi_dma_0"},
            "toolchain": {
                "vivado_version": "2022.2",
                "vitis_hls_version": "2022.2",
                "sha256": "c" * 64,
            },
            "artifacts": {
                name: dict(record)
                for name in ("checkpoint", "bitstream", "xsa", "hwh", "runtime")
            },
        }

        self.assertTrue(audit_zcu104_deployment_binding(payload)["overall_pass"])
        del payload["artifacts"]["xsa"]
        self.assertFalse(audit_zcu104_deployment_binding(payload)["overall_pass"])


def _write_capture(path: Path, power: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp_s", "power_w"])
        writer.writeheader()
        for second in range(61):
            writer.writerow({"timestamp_s": second, "power_w": power})


class Zcu104PowerContractTests(unittest.TestCase):
    def test_v2_power_manifest_binds_platform_thermal_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            idle = []
            active = []
            runtime_sha = "d" * 64
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
                            "board_serial": "36708",
                            "runtime_sha256": runtime_sha,
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
                "schema_version": 2,
                "measurement_source": "external_power_meter_csv",
                "rail_scope": "board_input_total",
                "pmbus_role": "auxiliary_only",
                "instrument": {
                    "model": "test",
                    "sample_rate_hz": 1,
                    "calibration": "traceable-test",
                    "calibration_certificate_id": "CERT-1",
                    "calibration_valid_at_measurement": True,
                },
                "time_sync": {"verified": True, "max_offset_ms": 1.0},
                "thermal": {"ambient_c": 25.0, "board_c": 40.0, "fan_mode": "fixed"},
                "platform_binding": {
                    "board_model": "ZCU104",
                    "board_serial": "36708",
                    "xsa_sha256": "b" * 64,
                    "hwh_sha256": "c" * 64,
                    "runtime_sha256": runtime_sha,
                    "toolchain_sha256": "e" * 64,
                    "vivado_version": "2022.2",
                    "vitis_hls_version": "2022.2",
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
        self.assertTrue(result["gates"]["receipt_platform_binding_match"])


if __name__ == "__main__":
    unittest.main()
