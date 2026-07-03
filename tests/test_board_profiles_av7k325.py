import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwnas_fpga.hardware import get_board_profile
from hwnas_fpga.runtime import build_hardware_spec


class AlinxAv7k325BoardProfileTests(unittest.TestCase):
    def test_aliases_resolve_to_alinx_av7k325_profile(self) -> None:
        by_short_name = get_board_profile("av7k325")
        by_full_name = get_board_profile("ALINX AV7K325")

        self.assertEqual(by_short_name.name, "ALINX AV7K325 (XC7K325T-2FFG900I)")
        self.assertEqual(by_full_name.name, by_short_name.name)

    def test_profile_matches_board_resources(self) -> None:
        profile = get_board_profile("alinx_av7k325")

        self.assertEqual(profile.clock_mhz, 200)
        self.assertEqual(profile.max_lut, 203_800)
        self.assertEqual(profile.max_ff, 407_600)
        self.assertEqual(profile.max_bram, 445)
        self.assertEqual(profile.max_dsp, 840)
        self.assertAlmostEqual(profile.memory_bandwidth_gbps, 12.8)
        self.assertEqual(profile.offchip_mem_mb, 2048.0)

    def test_runtime_builder_can_use_alinx_board_alias(self) -> None:
        hardware_spec = build_hardware_spec({"hardware": {"board": "av7k325"}})

        self.assertEqual(hardware_spec.name, "ALINX AV7K325 (XC7K325T-2FFG900I)")
        self.assertEqual(hardware_spec.max_dsp, 840)
        self.assertEqual(hardware_spec.offchip_mem_mb, 2048.0)


if __name__ == "__main__":
    unittest.main()
