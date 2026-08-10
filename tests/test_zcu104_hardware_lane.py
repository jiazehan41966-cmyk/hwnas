from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hls_lut_builder" / "scripts"))

from common import build_cases, load_config, materialize_case_project  # noqa: E402


CONFIG = (
    ROOT
    / "hls_lut_builder"
    / "configs"
    / "candidate_kernels_zcu104_cfem20_v1.yaml"
)


def test_cfem20_hardware_config_binds_target_and_all_frozen_cost_keys(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert raw["toolchain"]["part"] == "xczu7ev-ffvc1156-2-e"
    assert raw["toolchain"]["vitis_hls_bin"].endswith("Vitis_HLS/2022.2/bin/vitis_hls")
    assert raw["measurement_contract"]["search_space_version"] == "fourstage_cfem20_v1"
    assert raw["measurement_contract"]["source_binding_mode"] == "source_freeze_manifest"
    assert raw["measurement_contract"]["source_freeze_manifest"].endswith(
        "source_freeze_operator_cost_execution_v4_20260810/source_freeze_manifest.json"
    )
    assert "frozen_source_head" not in raw["measurement_contract"]

    operators = raw["operators"]
    assert len(operators) == 9
    assert {value["op_spec_defaults"]["cost_key"] for value in operators.values()} == {
        "stage2__k3_e3__16x112x112_to_24x56x56__s2",
        "stage2__k3_e6__16x112x112_to_24x56x56__s2",
        "stage2__k5_e3__16x112x112_to_24x56x56__s2",
        "stage2__k5_e6__16x112x112_to_24x56x56__s2",
        "stage4__skip__32x28x28_to_32x28x28__s1",
        "stage4__mbconv_k3_e3__32x28x28_to_32x28x28__s1",
        "stage4__mbconv_k5_e3__32x28x28_to_32x28x28__s1",
        "stage4__dmdc_conv_sonar__32x28x28_to_32x28x28__s1",
        "stage4__cfem_sonar__32x28x28_to_32x28x28__s1",
    }

    blocked = {
        name: value["blocked_reason"]
        for name, value in operators.items()
        if not value["enabled"]
    }
    assert set(blocked) == {"dmdc_conv_sonar_stage4", "cfem_sonar_stage4"}

    config = load_config(CONFIG)
    cases = build_cases(config, project_root_override=tmp_path)
    assert len(cases) == 7
    assert {case.op_spec["cost_key"] for case in cases} == {
        value["op_spec_defaults"]["cost_key"]
        for value in operators.values()
        if value["enabled"]
    }


def test_generated_existing_templates_are_zcu104_exact_shape_and_resolve(tmp_path):
    config = load_config(CONFIG)
    cases = build_cases(config, project_root_override=tmp_path)

    for case in cases:
        materialize_case_project(case)
        text = case.template_path.read_text(encoding="utf-8")
        assert "__TOP_FUNCTION__" in text
        assert "-Iinclude" in case.tcl_path.read_text(encoding="utf-8")
        assert (case.project_dir / "include" / "common_defs.h").is_file()
        assert (case.project_dir / "include" / "common_types.h").is_file()
        assert case.parameters["target_clock_ns"] == 5.0
        if case.op_spec["frozen_operator_id"].startswith("k"):
            assert case.parameters["feature_h"] == 112
            assert case.parameters["out_h"] == 56
        else:
            assert case.parameters["feature_h"] == 28
            assert case.parameters["out_h"] == 28
