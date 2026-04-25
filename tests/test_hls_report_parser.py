import tempfile
import unittest
from pathlib import Path

from hwnas_fpga.hardware import parse_hls_report, parse_hls_report_text


class HlsReportParserTests(unittest.TestCase):
    def test_parse_text_report(self) -> None:
        metrics = parse_hls_report_text(
            "\n".join(
                [
                    "Estimated Clock Period: 4.500",
                    "Latency (cycles): 12345",
                    "Latency (ns): 61725.0",
                    "Initiation Interval: 1",
                    "Pipeline Depth: 32",
                    "BRAM_18K | 12",
                    "BRAM_36K | 3",
                    "DSP48E   | 34",
                    "FF       | 5678",
                    "LUT      | 4321",
                ]
            )
        )
        self.assertEqual(metrics["cycles"], 12345)
        self.assertEqual(metrics["dsp"], 34)
        self.assertEqual(metrics["bram"], 12)
        self.assertEqual(metrics["bram_36k"], 3)
        self.assertEqual(metrics["ii"], 1)
        self.assertEqual(metrics["depth"], 32)
        self.assertAlmostEqual(metrics["fmax_est_mhz"], 222.222, places=2)
        self.assertEqual(metrics["lut"], 4321)
        self.assertEqual(metrics["report_format"], "text")

    def test_parse_xml_report(self) -> None:
        xml_text = """\
<profile>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>5.000</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Best-caseLatency>100</Best-caseLatency>
      <Average-caseLatency>120</Average-caseLatency>
      <Worst-caseLatency>140</Worst-caseLatency>
      <Interval-max>2</Interval-max>
      <PipelineDepth>18</PipelineDepth>
      <Worst-caseRealTimeLatency>700.0 ns</Worst-caseRealTimeLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>4</BRAM_18K>
      <BRAM_36K>1</BRAM_36K>
      <DSP>8</DSP>
      <FF>1024</FF>
      <LUT>2048</LUT>
    </Resources>
  </AreaEstimates>
</profile>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "csynth.xml"
            report_path.write_text(xml_text, encoding="utf-8")
            metrics = parse_hls_report(report_path)

        self.assertEqual(metrics["cycles"], 140)
        self.assertEqual(metrics["latency_ns"], 700.0)
        self.assertEqual(metrics["dsp"], 8)
        self.assertEqual(metrics["bram"], 4)
        self.assertEqual(metrics["bram_36k"], 1)
        self.assertEqual(metrics["ii"], 2)
        self.assertEqual(metrics["depth"], 18)
        self.assertEqual(metrics["fmax_est_mhz"], 200.0)
        self.assertEqual(metrics["lut"], 2048)
        self.assertEqual(metrics["report_format"], "xml")

    def test_parse_xml_report_with_ms_realtime_latency(self) -> None:
        xml_text = """\
<profile>
  <PerformanceEstimates>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>3.650</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Worst-caseLatency>4014104</Worst-caseLatency>
      <Interval-max>10</Interval-max>
      <PipelineDepth>33</PipelineDepth>
      <Worst-caseRealTimeLatency>20.071 ms</Worst-caseRealTimeLatency>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>890</BRAM_18K>
      <DSP>840</DSP>
      <FF>407600</FF>
      <LUT>203800</LUT>
    </Resources>
  </AreaEstimates>
</profile>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "csynth.xml"
            report_path.write_text(xml_text, encoding="utf-8")
            metrics = parse_hls_report(report_path)

        self.assertEqual(metrics["cycles"], 4014104)
        self.assertEqual(metrics["latency_ns"], 20_071_000.0)
        self.assertEqual(metrics["ii"], 10)
        self.assertEqual(metrics["depth"], 33)
        self.assertEqual(metrics["dsp"], 840)
        self.assertEqual(metrics["bram"], 890)
        self.assertEqual(metrics["lut"], 203800)

    def test_parse_xml_report_prefers_used_resources_and_enriches_pipeline_from_log(self) -> None:
        xml_text = """\
<profile>
  <PerformanceEstimates>
    <PipelineType>no</PipelineType>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>5.596</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Worst-caseLatency>853378</Worst-caseLatency>
      <Worst-caseRealTimeLatency>4.776 ms</Worst-caseRealTimeLatency>
      <Interval-max>853379</Interval-max>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>140</BRAM_18K>
      <DSP>151</DSP>
      <FF>17749</FF>
      <LUT>16205</LUT>
    </Resources>
    <AvailableResources>
      <BRAM_18K>890</BRAM_18K>
      <DSP>840</DSP>
      <FF>407600</FF>
      <LUT>203800</LUT>
    </AvailableResources>
  </AreaEstimates>
</profile>
"""
        log_text = """\
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 78, loop 'VITIS_LOOP_66_6_VITIS_LOOP_67_7_VITIS_LOOP_68_8'
INFO: [HLS 200-790] **** Loop Constraint Status: All loop constraints were satisfied.
INFO: [HLS 200-789] **** Estimated Fmax: 178.70 MHz
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_root = Path(tmpdir) / "pilot_case"
            report_path = case_root / "project" / "solution1" / "syn" / "report" / "stem_csynth.xml"
            log_path = case_root / "logs" / "vitis_hls.log"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(xml_text, encoding="utf-8")
            log_path.write_text(log_text, encoding="utf-8")

            metrics = parse_hls_report(report_path)

        self.assertEqual(metrics["cycles"], 853378)
        self.assertEqual(metrics["latency_ns"], 4_776_000.0)
        self.assertEqual(metrics["ii"], 1)
        self.assertEqual(metrics["depth"], 78)
        self.assertEqual(metrics["dsp"], 151)
        self.assertEqual(metrics["bram"], 140)
        self.assertEqual(metrics["ff"], 17749)
        self.assertEqual(metrics["lut"], 16205)
        self.assertAlmostEqual(metrics["fmax_est_mhz"], 178.699, places=2)

    def test_parse_xml_report_keeps_zero_bram_usage(self) -> None:
        xml_text = """\
<profile>
  <PerformanceEstimates>
    <PipelineType>no</PipelineType>
    <SummaryOfTimingAnalysis>
      <EstimatedClockPeriod>5.877</EstimatedClockPeriod>
    </SummaryOfTimingAnalysis>
    <SummaryOfOverallLatency>
      <Worst-caseLatency>51791</Worst-caseLatency>
      <Worst-caseRealTimeLatency>0.304 ms</Worst-caseRealTimeLatency>
      <Interval-max>51792</Interval-max>
    </SummaryOfOverallLatency>
  </PerformanceEstimates>
  <AreaEstimates>
    <Resources>
      <BRAM_18K>0</BRAM_18K>
      <DSP>192</DSP>
      <FF>12044</FF>
      <LUT>11509</LUT>
    </Resources>
    <AvailableResources>
      <BRAM_18K>890</BRAM_18K>
      <DSP>840</DSP>
      <FF>407600</FF>
      <LUT>203800</LUT>
    </AvailableResources>
  </AreaEstimates>
</profile>
"""
        log_text = """\
INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 1, Depth = 20, loop 'VITIS_LOOP_65_9_VITIS_LOOP_66_10'
INFO: [HLS 200-789] **** Estimated Fmax: 170.15 MHz
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            case_root = Path(tmpdir) / "packed_case"
            report_path = case_root / "project" / "solution1" / "syn" / "report" / "stem_csynth.xml"
            log_path = case_root / "logs" / "vitis_hls.log"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(xml_text, encoding="utf-8")
            log_path.write_text(log_text, encoding="utf-8")

            metrics = parse_hls_report(report_path)

        self.assertEqual(metrics["cycles"], 51791)
        self.assertEqual(metrics["latency_ns"], 304000.0)
        self.assertEqual(metrics["ii"], 1)
        self.assertEqual(metrics["depth"], 20)
        self.assertEqual(metrics["bram"], 0)
        self.assertEqual(metrics["bram_18k"], 0)
        self.assertEqual(metrics["dsp"], 192)
        self.assertEqual(metrics["ff"], 12044)
        self.assertEqual(metrics["lut"], 11509)


if __name__ == "__main__":
    unittest.main()
