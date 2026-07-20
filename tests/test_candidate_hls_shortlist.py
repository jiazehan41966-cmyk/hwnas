import tempfile
import unittest
from pathlib import Path

from scripts.run_candidate_hls_shortlist import (
    dominates,
    generate_missing_config,
)


class ParetoTests(unittest.TestCase):
    def test_hardware_dominance(self) -> None:
        better = {
            "aggregate_hls": {
                "latency_ms": 1,
                "dsp": 2,
                "lut": 3,
                "bram": 4,
            }
        }
        worse = {
            "aggregate_hls": {
                "latency_ms": 2,
                "dsp": 2,
                "lut": 4,
                "bram": 5,
            }
        }
        self.assertTrue(dominates(better, worse))
        self.assertFalse(dominates(worse, better))


class ConfigGenerationTests(unittest.TestCase):
    def test_generates_one_unique_case_per_op_key(self) -> None:
        role = {
            "role": "stage0_block0",
            "op_spec": {
                "op": "mbconv_e1_k3",
                "kernel_size": 3,
                "in_channels": 8,
                "out_channels": 8,
                "stride": 1,
                "groups": 1,
                "expand_ratio": 1,
                "input_resolution": [16, 16],
                "bitwidth": 8,
                "input_parallelism": 1,
                "output_parallelism": 1,
                "unroll_factor": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_missing_config([role, role], output_dir=Path(tmpdir))
            text = path.read_text(encoding="utf-8")
            self.assertEqual(text.count("candidate_case_0000:"), 1)


if __name__ == "__main__":
    unittest.main()
