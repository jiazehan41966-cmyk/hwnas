import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "hls_lut_builder" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
BOARD_SCRIPT_ROOT = REPO_ROOT / "hls_lut_builder" / "board_harness" / "scripts"
if str(BOARD_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOARD_SCRIPT_ROOT))

from common import _build_parameters, build_cases, load_config, load_template, render_template  # noqa: E402
from generate_harness import compute_case_meta, parse_constants  # noqa: E402
from sonar_classifier_e2e_board_validation import axil_completion_latch_text  # noqa: E402


CONFIG_PATH = REPO_ROOT / "hls_lut_builder" / "configs" / "candidate_kernels.yaml"
LATENCY_SCOPE_PATH = REPO_ROOT / "hls_lut_builder" / "LATENCY_SCOPE.md"
KERNEL_SPEC_PATH = REPO_ROOT / "hls_lut_builder" / "docs" / "kernel_specification.md"
COMMON_TYPES_PATH = REPO_ROOT / "hls_lut_builder" / "include" / "common_types.h"


class HlsTemplateContractTests(unittest.TestCase):
    def test_core_operator_templates_follow_packed_stream_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        operators = config["operators"]
        core_names = [
            "stem_conv_k3_s2",
            "dw_conv_k3",
            "dw_conv_k5",
            "pw_conv",
            "mbconv_e3_k3",
            "mbconv_e3_k5",
            "mbconv_e6_k3",
            "mbconv_e6_k5",
            "skip",
            "global_avg_pool",
            "fc_layer",
        ]

        for operator_name in core_names:
            with self.subTest(operator=operator_name):
                operator_cfg = operators[operator_name]
                template_path = (CONFIG_PATH.parent / operator_cfg["template"]).resolve()
                text = template_path.read_text(encoding="utf-8")
                self.assertIn('#include "common_defs.h"', text)
                self.assertNotIn('#include "common_types.h"', text)
                self.assertIn("DECLARE_AXIS_INTERFACE()", text)
                self.assertIn("DECLARE_BRAM_WEIGHTS()", text)
                self.assertIn("DECLARE_CONTROL_INTERFACE()", text)
                self.assertIn("hls::stream<", text)
                self.assertIn("MAKE_AXIS_TYPE(", text)
                self.assertNotIn("read_stream_tensor<", text)
                self.assertNotIn("write_stream_tensor<", text)

    def test_extension_operator_templates_follow_packed_stream_contract(self) -> None:
        config = load_config(CONFIG_PATH)
        operators = config["operators"]
        extension_names = [
            "fused_mbconv_e3_k3",
            "fused_mbconv_e6_k3",
            "mixconv",
            "denoise",
            "edge",
        ]

        for operator_name in extension_names:
            with self.subTest(operator=operator_name):
                operator_cfg = operators[operator_name]
                template_path = (CONFIG_PATH.parent / operator_cfg["template"]).resolve()
                text = template_path.read_text(encoding="utf-8")
                self.assertIn('#include "common_defs.h"', text)
                self.assertNotIn('#include "common_types.h"', text)
                self.assertIn("DECLARE_AXIS_INTERFACE()", text)
                self.assertIn("DECLARE_BRAM_WEIGHTS()", text)
                self.assertIn("DECLARE_CONTROL_INTERFACE()", text)
                self.assertIn("hls::stream<", text)
                self.assertIn("MAKE_AXIS_TYPE(", text)
                self.assertNotIn("read_stream_tensor<", text)
                self.assertNotIn("write_stream_tensor<", text)

    def test_mixconv_v2_template_has_versioned_integer_contract(self) -> None:
        template = (
            REPO_ROOT / "hls_lut_builder" / "templates" / "mixconv_v2.cpp.tmpl"
        ).read_text(encoding="utf-8")
        self.assertIn("static constexpr int BRANCH3_CH = IN_CH / 2;", template)
        self.assertIn("DW3_WEIGHT_SIZE +", template)
        self.assertIn("requantize_nearest_even", template)
        self.assertIn("dw_multiplier_numerators", template)
        self.assertIn("pw_multiplier_numerators", template)
        self.assertIn("value += static_cast<acc_t>(residual_channels", template)
        self.assertNotIn("relu6_quant(acc_tile", template)
        project_requant = template.index("requantize_nearest_even(", template.index("project_stage"))
        residual_add = template.index("value += static_cast<acc_t>(residual_channels")
        self.assertLess(project_requant, residual_add)

        config_path = (
            REPO_ROOT
            / "hls_lut_builder"
            / "configs"
            / "candidate_kernels_mixconv_v2_stage32.yaml"
        )
        cases = build_cases(load_config(config_path))
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.parameters["in_channels"], 32)
        self.assertEqual(case.parameters["out_channels"], 32)
        self.assertEqual(case.parameters["feature_h"], 28)
        source = render_template(
            load_template(case.template_path),
            {"top_function": case.top_function, **case.parameters},
        )
        self.assertNotIn("__", source)
        self.assertIn("static constexpr int BRANCH3_CH = IN_CH / 2;", source)

    def test_docs_and_headers_state_same_packed_stream_policy(self) -> None:
        latency_scope = LATENCY_SCOPE_PATH.read_text(encoding="utf-8")
        kernel_spec = KERNEL_SPEC_PATH.read_text(encoding="utf-8")
        common_types = COMMON_TYPES_PATH.read_text(encoding="utf-8")

        self.assertIn("packed_stream_v1", latency_scope)
        self.assertIn("packed_stream_v1", kernel_spec)
        self.assertIn("feature input", latency_scope)
        self.assertIn("feature input: packed `AXI-Stream`", kernel_spec)
        self.assertIn("common_types.h", kernel_spec)
        self.assertIn("deprecated", common_types.lower())

    def test_wide_pw_conv_uses_segmented_axis_words(self) -> None:
        config = load_config(CONFIG_PATH)
        cases = build_cases(
            config,
            operator_filter=["pw_conv"],
            case_filter=["pw_conv_pw_head_7_320_1280_baseline_pi1_po1_u1_main_5ns"],
        )
        self.assertEqual(len(cases), 1)
        case = cases[0]

        source = render_template(
            load_template(case.template_path),
            {"top_function": case.top_function, **case.parameters},
        )
        constants = parse_constants(source)
        case_meta = compute_case_meta(case.case_name, constants)

        self.assertEqual(constants["MAX_AXIS_BITS"], 4096)
        self.assertEqual(constants["OUT_PACK_CH"], 512)
        self.assertEqual(constants["OUTPUT_PACKETS_PER_PIXEL"], 3)
        self.assertLessEqual(constants["OUT_PACK_CH"] * 8, constants["MAX_AXIS_BITS"])
        self.assertEqual(case_meta["input_word_count"], 7 * 7)
        self.assertEqual(case_meta["output_word_count"], 7 * 7 * 3)

    def test_late_mbconv_segments_expanded_internal_stream(self) -> None:
        config = load_config(CONFIG_PATH)
        cases = build_cases(
            config,
            operator_filter=["mbconv_e6_k3"],
            case_filter=["mbconv_e6_k3_mbconv_stage7_7_160_320_s1_baseline_pi1_po1_u1_main_5ns"],
        )
        self.assertEqual(len(cases), 1)
        case = cases[0]

        source = render_template(
            load_template(case.template_path),
            {"top_function": case.top_function, **case.parameters},
        )
        constants = parse_constants(source)

        self.assertEqual(constants["EXP_CH"], 960)
        self.assertEqual(constants["EXP_PACK_CH"], 512)
        self.assertEqual(constants["EXP_PACKETS_PER_PIXEL"], 2)
        self.assertLessEqual(constants["EXP_PACK_CH"] * 8, constants["MAX_AXIS_BITS"])
        self.assertIn("pack_channel_segment<EXP_CH, EXP_PACK_CH>", source)
        self.assertIn("unpack_packed_channel_segment<EXP_CH, EXP_PACK_CH>", source)

    def test_residual_fifo_covers_depthwise_line_buffer_prefill(self) -> None:
        parameters = _build_parameters(
            defaults={},
            operator_name="mbconv_e1_k3",
            operator_cfg={
                "parameters": {
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "in_channels": 12,
                    "out_channels": 12,
                    "expand_ratio": 1,
                }
            },
            shape_name="stage3_block1_14x14x12",
            shape_cfg={"feature_h": 14, "feature_w": 14},
            implementation_name="baseline",
            implementation_cfg={},
            clock_profile_name="main_5ns",
            clock_cfg={"period_ns": 5.0, "target_clock_mhz": 200.0},
        )

        self.assertEqual(parameters["residual_stream_depth"], 16)
        rendered = render_template(
            "#pragma HLS STREAM variable=residual_stream depth=__RESIDUAL_STREAM_DEPTH__",
            parameters,
        )
        self.assertEqual(rendered, "#pragma HLS STREAM variable=residual_stream depth=16")

    def test_axil_completion_latches_independent_done_pulses(self) -> None:
        text = axil_completion_latch_text([{"role": "first"}, {"role": "second"}])

        self.assertIn("wire [1:0] axil_done_now = {second_axil_done, first_axil_done};", text)
        self.assertIn("axil_done_seen <= axil_done_seen | axil_done_now;", text)
        self.assertIn("else if (start_pulse)", text)
        self.assertIn("wire all_axil_done = &axil_done_seen;", text)


if __name__ == "__main__":
    unittest.main()
