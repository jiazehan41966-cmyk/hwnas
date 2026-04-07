import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SearchSpaceProbeCliTests(unittest.TestCase):
    def test_probe_cli_generates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "probe.yaml"
            output_dir = tmp_path / "results"
            config_path.write_text(
                """
project:
  name: probe-smoke
  output_dir: results
dataset:
  name: dummy
  input_channels: 1
  image_size: 64
  num_classes: 4
search_space:
  family_profile: lightweight_sonar
constraints:
  max_latency_ms: 50.0
  max_dsp: 220
  max_bram: 140
  max_lut: 53200
hardware:
  board: zynq7020
  quantization_bits: 8
                """.strip(),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "run_search_space_probe.py"),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--run-name",
                    "probe-smoke",
                    "--num-samples",
                    "8",
                ],
                check=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )

            summary_path = output_dir / "probe-smoke" / "results" / "probe_summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["num_samples"], 8)
            self.assertIn("summary", summary)
            self.assertEqual(summary["summary"]["total_samples"], 8)


if __name__ == "__main__":
    unittest.main()
