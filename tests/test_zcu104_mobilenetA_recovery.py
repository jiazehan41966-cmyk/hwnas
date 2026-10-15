from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
HLS_SCRIPTS = ROOT / "hls_lut_builder" / "scripts"
if str(HLS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(HLS_SCRIPTS))

from common import build_cases, load_config, load_template, render_template


CONFIG = ROOT / "hls_lut_builder" / "configs" / "candidate_kernels_zcu104_mobilenetA_recovery_v2.yaml"


def test_recovery_config_is_exactly_the_13_blocked_rows() -> None:
    cases = build_cases(load_config(CONFIG), sort_cases=True)
    assert len(cases) == 13
    assert {case.operator for case in cases} == {
        "dw_conv_k3_recovery",
        "dw_conv_k5_recovery",
        "pw_conv_recovery",
    }
    assert sum(case.operator == "dw_conv_k3_recovery" for case in cases) == 3
    assert sum(case.operator == "dw_conv_k5_recovery" for case in cases) == 8
    assert sum(case.operator == "pw_conv_recovery" for case in cases) == 2
    assert all(case.part == "xczu7ev-ffvc1156-2-e" for case in cases)
    assert all(case.target_clock_mhz == 200.0 for case in cases)
    assert all(case.implementation_name == "lowresource_p4_cb4_ii8" for case in cases)


def test_recovery_templates_render_without_placeholders() -> None:
    cases = build_cases(load_config(CONFIG), sort_cases=True)
    for case in cases:
        rendered = render_template(
            load_template(case.template_path),
            {"top_function": case.top_function, **case.parameters},
        )
        assert "__" not in rendered
        assert case.top_function in rendered
        assert "ap_int<8>" not in rendered or "common_defs.h" in rendered


def test_recovery_workspace_is_isolated_from_v1() -> None:
    cases = build_cases(load_config(CONFIG), sort_cases=True)
    assert all("mobilenetA_full_dmdc_cfem_v2_recovered" in str(case.project_dir) for case in cases)
    assert all("mobilenetA_full_dmdc_cfem_v1" not in str(case.project_dir) for case in cases)
