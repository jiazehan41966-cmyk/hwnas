import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "hls_lut_builder" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import run_synthesis  # noqa: E402


class RunSynthesisCliTests(unittest.TestCase):
    def test_parse_args_supports_downstream_check(self) -> None:
        argv = [
            "run_synthesis.py",
            "--config",
            "hls_lut_builder/configs/candidate_kernels.yaml",
            "--downstream-check",
            "--vivado",
            r"F:\vivado\Vivado\2023.2\bin\vivado.bat",
            "--downstream-timeout-minutes",
            "90",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = run_synthesis.parse_args()

        self.assertTrue(args.downstream_check)
        self.assertEqual(args.vivado, r"F:\vivado\Vivado\2023.2\bin\vivado.bat")
        self.assertEqual(args.downstream_timeout_minutes, 90)


if __name__ == "__main__":
    unittest.main()
