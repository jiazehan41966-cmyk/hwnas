import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from hwnas_fpga.hardware import (
    LutEntry,
    LutTable,
    OpSpec,
    build_lut_from_manifest,
    get_board_profile,
    parse_hls_report_text,
)
from hwnas_fpga.runtime import load_lut_query_engine


class BoardProfileTests(unittest.TestCase):
    def test_zynq7020_profile_contains_memory_fields(self) -> None:
        profile = get_board_profile("zynq7020")
        self.assertEqual(profile.max_dsp, 220)
        self.assertEqual(profile.max_bram, 140)
        self.assertGreater(profile.memory_bandwidth_gbps, 0.0)
        self.assertGreater(profile.offchip_mem_mb, 0.0)


class LutRuntimeTests(unittest.TestCase):
    def test_load_lut_query_engine_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lut_path = Path(tmpdir) / "fpga_lut.pkl"
            table = LutTable(
                [
                    LutEntry(
                        op_spec=OpSpec(
                            op="conv",
                            kernel_size=3,
                            in_channels=16,
                            out_channels=16,
                            input_resolution=(56, 56),
                        ),
                        latency_ms=0.1,
                        cycles=20_000,
                        dsp=8,
                        bram=2,
                        lut=128,
                        power_w=3.0,
                        energy_mj=0.3,
                    )
                ]
            )
            table.save(str(lut_path))
            engine = load_lut_query_engine({"hardware": {"lut_path": str(lut_path)}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="conv",
                        kernel_size=3,
                        in_channels=16,
                        out_channels=16,
                        input_resolution=(56, 56),
                    )
                )
            )

    def test_parse_hls_report_text(self) -> None:
        report_text = """
        == Performance Estimates
        Latency (cycles): 12345
        == Utilization Estimates
        BRAM_18K | 12
        DSP48E   | 34
        FF       | 5678
        LUT      | 4321
        """
        metrics = parse_hls_report_text(report_text)
        self.assertEqual(metrics["cycles"], 12345)
        self.assertEqual(metrics["bram"], 12)
        self.assertEqual(metrics["dsp"], 34)
        self.assertEqual(metrics["lut"], 4321)

    def test_build_lut_from_manifest_with_relative_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "reports" / "conv.rpt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                """
                Latency (cycles): 5000
                BRAM_18K | 4
                DSP48E   | 12
                LUT      | 345
                Power    | 1.25
                """,
                encoding="utf-8",
            )
            manifest_path = tmp_path / "lut_manifest.yaml"
            manifest_path.write_text(
                """
clock_mhz: 200
entries:
  - op: conv
    kernel_size: 3
    in_channels: 16
    out_channels: 32
    stride: 1
    input_resolution: [56, 56]
    report: reports/conv.rpt
                """.strip(),
                encoding="utf-8",
            )

            table, summary = build_lut_from_manifest(manifest_path)
            self.assertEqual(summary["entries_built"], 1)

            entry = table.query(
                OpSpec(
                    op="conv",
                    kernel_size=3,
                    in_channels=16,
                    out_channels=32,
                    stride=1,
                    input_resolution=(56, 56),
                )
            )
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertEqual(entry.cycles, 5000)
            self.assertAlmostEqual(entry.latency_ms, 0.025, places=6)
            self.assertEqual(entry.dsp, 12)
            self.assertEqual(entry.bram, 4)
            self.assertEqual(entry.lut, 345)

    def test_run_build_lut_cli_generates_pickle_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            report_path = tmp_path / "mbconv.rpt"
            report_path.write_text(
                """
                Latency (cycles): 12000
                BRAM_18K | 6
                DSP48E   | 24
                LUT      | 789
                Power    | 2.5
                """,
                encoding="utf-8",
            )
            manifest_path = tmp_path / "lut_manifest.json"
            manifest_path.write_text(
                """
{
  "clock_mhz": 200,
  "entries": [
    {
      "op": "mbconv",
      "kernel_size": 3,
      "in_channels": 32,
      "out_channels": 64,
      "stride": 1,
      "expand_ratio": 4,
      "input_resolution": [28, 28],
      "report": "mbconv.rpt"
    }
  ]
}
                """.strip(),
                encoding="utf-8",
            )
            output_path = tmp_path / "fpga_lut.pkl"
            summary_path = tmp_path / "summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "run_build_lut.py"),
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output_path),
                    "--summary-json",
                    str(summary_path),
                ],
                check=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            engine = load_lut_query_engine({"hardware": {"lut_path": str(output_path)}})
            self.assertIsNotNone(engine)
            self.assertIsNotNone(
                engine.query(
                    OpSpec(
                        op="mbconv",
                        kernel_size=3,
                        in_channels=32,
                        out_channels=64,
                        stride=1,
                        expand_ratio=4,
                        input_resolution=(28, 28),
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
