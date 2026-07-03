import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "hls_lut_builder" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import build_cases, load_config, select_pilot_cases  # noqa: E402


class HlsBuilderPilotTests(unittest.TestCase):
    def test_select_pilot_cases_picks_one_case_per_enabled_core_operator(self) -> None:
        config = load_config(REPO_ROOT / "hls_lut_builder" / "configs" / "candidate_kernels.yaml")
        cases = build_cases(config, sort_cases=False)
        pilot_cases = select_pilot_cases(cases)
        self.assertEqual(
            [case.case_name for case in pilot_cases],
            [
                "stem_conv_k3_s2_stem_224_1_32_s2_baseline_pi1_po1_u1_main_5ns",
                "dw_conv_k3_dw_ir16_e1_112_c32_s1_baseline_pi1_po1_u1_main_5ns",
                "dw_conv_k5_dw_ir16_e1_112_c32_s1_baseline_pi1_po1_u1_main_5ns",
                "pw_conv_pw_ir16_proj_112_32_16_baseline_pi1_po1_u1_main_5ns",
                "mbconv_e3_k3_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
                "mbconv_e3_k5_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
                "mbconv_e6_k3_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
                "mbconv_e6_k5_mbconv_stage2_112_16_24_s2_baseline_pi1_po1_u1_main_5ns",
                "skip_skip_56_24_baseline_pi1_po1_u1_main_5ns",
                "global_avg_pool_gap_7_320_baseline_pi1_po1_u1_main_5ns",
                "fc_layer_fc_1_1280_8_baseline_pi1_po1_u1_main_5ns",
            ],
        )


if __name__ == "__main__":
    unittest.main()
