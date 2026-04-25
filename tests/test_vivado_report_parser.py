import unittest

from hwnas_fpga.hardware import (
    approximate_fmax_from_wns,
    parse_vivado_timing_summary_text,
    parse_vivado_utilization_text,
)


class VivadoReportParserTests(unittest.TestCase):
    def test_parse_vivado_utilization_text(self) -> None:
        report_text = """\
+-------------------+------+-------+------------+-----------+-------+
|     Site Type     | Used | Fixed | Prohibited | Available | Util% |
+-------------------+------+-------+------------+-----------+-------+
| Slice LUTs        | 2512 |     0 |          0 |    203800 |  1.23 |
| Slice Registers   | 3075 |     0 |          0 |    407600 |  0.75 |
| Block RAM Tile    |  4.5 |     0 |          0 |       445 |  1.01 |
|   RAMB36/FIFO*    |    0 |     0 |          0 |       445 |  0.00 |
|   RAMB18          |    9 |     0 |          0 |       890 |  1.01 |
| DSPs              |  335 |     0 |          0 |       840 | 39.88 |
"""
        metrics = parse_vivado_utilization_text(report_text)
        self.assertEqual(metrics["lut"], 2512)
        self.assertEqual(metrics["ff"], 3075)
        self.assertEqual(metrics["block_ram_tile"], 4.5)
        self.assertEqual(metrics["ramb36"], 0)
        self.assertEqual(metrics["ramb18"], 9)
        self.assertEqual(metrics["dsp"], 335)

    def test_parse_vivado_timing_summary_text(self) -> None:
        report_text = """\
Setup :            0  Failing Endpoints,  Worst Slack        0.075ns,  Total Violation        0.000ns
Hold  :            0  Failing Endpoints,  Worst Slack        0.129ns,  Total Violation        0.000ns
Slack (MET) :             0.075ns  (required time - arrival time)
  Requirement:            5.000ns  (ap_clk rise@5.000ns - ap_clk rise@0.000ns)
  Data Path Delay:        4.841ns  (logic 0.223ns (4.606%)  route 4.618ns (95.394%))
"""
        metrics = parse_vivado_timing_summary_text(report_text)
        self.assertAlmostEqual(metrics["setup_wns_ns"], 0.075, places=3)
        self.assertAlmostEqual(metrics["hold_whs_ns"], 0.129, places=3)
        self.assertAlmostEqual(metrics["target_clock_ns"], 5.0, places=3)
        self.assertAlmostEqual(metrics["data_path_delay_ns"], 4.841, places=3)
        self.assertAlmostEqual(metrics["logic_delay_ns"], 0.223, places=3)
        self.assertAlmostEqual(metrics["route_delay_ns"], 4.618, places=3)
        self.assertAlmostEqual(metrics["fmax_est_mhz"], 203.046, places=3)

    def test_parse_vivado_timing_summary_text_single_endpoint(self) -> None:
        report_text = """\
Setup :            1  Failing Endpoint ,  Worst Slack       -0.002ns,  Total Violation       -0.002ns
Hold  :            0  Failing Endpoints,  Worst Slack        0.018ns,  Total Violation        0.000ns
Slack (VIOLATED) :        -0.002ns  (required time - arrival time)
  Requirement:            5.000ns  (ap_clk rise@5.000ns - ap_clk rise@0.000ns)
  Data Path Delay:        4.987ns  (logic 0.347ns (6.958%)  route 4.640ns (93.042%))
"""
        metrics = parse_vivado_timing_summary_text(report_text)
        self.assertAlmostEqual(metrics["setup_wns_ns"], -0.002, places=3)
        self.assertAlmostEqual(metrics["hold_whs_ns"], 0.018, places=3)
        self.assertAlmostEqual(metrics["target_clock_ns"], 5.0, places=3)
        self.assertAlmostEqual(metrics["data_path_delay_ns"], 4.987, places=3)
        self.assertAlmostEqual(metrics["logic_delay_ns"], 0.347, places=3)
        self.assertAlmostEqual(metrics["route_delay_ns"], 4.640, places=3)
        self.assertAlmostEqual(metrics["fmax_est_mhz"], 199.920, places=3)

    def test_approximate_fmax_from_negative_wns(self) -> None:
        self.assertAlmostEqual(
            approximate_fmax_from_wns(target_clock_ns=5.0, wns_ns=-0.250),
            190.476,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
