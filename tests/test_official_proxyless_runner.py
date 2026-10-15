from hwnas_fpga.search.official_proxyless_runner import _resolve_official_settings


def _base_config(target_hardware=None):
    proxyless = {
        "evaluator_mode": "weight_sharing_supernet",
        "comparison_boundary": "not_equal_to_independent_candidate_train_budget",
    }
    if target_hardware is not None:
        proxyless["target_hardware"] = target_hardware
    return {
        "dataset": {"num_workers": 0},
        "training": {"batch_size": 4},
        "hardware": {
            "operator_cost_lut_path": "artifacts/zcu104_full_campaign_v1/operator_cost_lut/mobilenetA_full_dmdc_cfem_v2_recovered/hardware_cost_lut.formal.json",
        },
        "search_space": {
            "stage_block_choices": [
                [{"op": "conv", "kernel_size": 1, "expand_ratio": 1, "stride": 1}],
                [
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2},
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 6, "stride": 2},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3, "stride": 2},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 6, "stride": 2},
                ],
                [{"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 2}],
                [
                    {"op": "skip", "kernel_size": 1, "expand_ratio": 1, "stride": 1},
                    {"op": "mbconv", "kernel_size": 3, "expand_ratio": 3, "stride": 1},
                    {"op": "mbconv", "kernel_size": 5, "expand_ratio": 3, "stride": 1},
                    {"op": "dmdc_conv_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1},
                    {"op": "cfem_sonar", "kernel_size": 3, "expand_ratio": 1, "stride": 1},
                ],
            ],
        },
        "search": {
            "episodes": 2,
            "eval_epochs": 3,
            "proxyless": proxyless,
        },
    }


def test_zcu104_proxyless_disables_reference_mobile_latency_by_default():
    settings = _resolve_official_settings(_base_config())

    assert settings["target_hardware"] is None
    assert settings["ref_value"] is None
    assert settings["evaluator_mode"] == "weight_sharing_supernet"
    assert settings["hardware_regularization_note"]["uses_zcu104_formal_lut"] is True
    assert (
        settings["hardware_regularization_note"]["official_mobile_latency_regularization"]
        == "disabled"
    )
    assert settings["first_block_as_stage0"] is True
    assert settings["first_block_candidates"] == ["1x1_Conv"]
    assert settings["bridge_skip_fixed_first_block"] is False
    assert settings["super_width_stages"] == [24, 32, 32]
    assert settings["super_stride_stages"] == [2, 2, 1]
    assert settings["conv_candidates"] == [
        ["3x3_MBConv3", "3x3_MBConv6", "5x5_MBConv3", "5x5_MBConv6"],
        ["3x3_MBConv3"],
        ["3x3_MBConv3", "5x5_MBConv3"],
    ]
    assert [row["choice"]["op"] for row in settings["unsupported_stage_block_choices"]] == [
        "dmdc_conv_sonar",
        "cfem_sonar",
    ]
    assert settings["warmup_epochs"] == 3
    assert settings["n_epochs"] == 2


def test_zcu104_proxyless_does_not_remap_latency_ms_to_mobile_lut():
    settings = _resolve_official_settings(_base_config("latency_ms"))

    assert settings["target_hardware"] is None
    assert settings["hardware_regularization_note"]["requested_target_hardware"] == "latency_ms"
    assert settings["hardware_regularization_note"]["resolved_target_hardware"] is None


def test_zcu104_proxyless_does_not_use_mobile_lut_even_if_requested():
    settings = _resolve_official_settings(_base_config("mobile"))

    assert settings["target_hardware"] is None
    assert settings["ref_value"] is None
    assert settings["hardware_regularization_note"]["requested_target_hardware"] == "mobile"
