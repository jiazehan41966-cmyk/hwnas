from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT_HLS = ROOT / "hardware" / "zcu104_dma_shell_v1" / "hls" / "mbconv_k3_e3"
PILOT_BD = (
    ROOT
    / "hardware"
    / "zcu104_dma_shell_v1"
    / "vivado"
    / "build_mbconv_k3_e3_pilot_bd.tcl"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_math_source_is_frozen_with_declared_interface_schedule_adaptations():
    identity = json.loads(
        (PILOT_HLS / "provenance" / "source_identity.json").read_text(encoding="utf-8")
    )
    source_path = PILOT_HLS / "src" / "zcu104_stage4_k3_e3_kernel.cpp"
    source = source_path.read_text(encoding="utf-8")
    assert sha256(source_path.read_bytes()) == identity["pilot_source_sha256"]
    assert source.count("#pragma HLS INTERFACE ap_ctrl_hs port=return") == 1
    assert source.count("static void load_parameters(") == 1
    assert source.count("#pragma HLS INLINE off") == 1
    assert identity["allowed_storage_interface_constraint"]["mathematical_effect"] == "none"
    assert identity["allowed_schedule_only_change"]["mathematical_effect"] == "none"


def test_operator_shape_types_and_parameter_layout_are_unchanged():
    source = (PILOT_HLS / "src" / "zcu104_stage4_k3_e3_kernel.cpp").read_text(
        encoding="utf-8"
    )
    for token in (
        "IN_CH = 32",
        "EXP_CH = 96",
        "OUT_CH = 32",
        "IN_H = 28",
        "IN_W = 28",
        "K = 3",
        "STRIDE = 1",
        "WEIGHT_SIZE = EXPAND_WEIGHT_SIZE + DEPTHWISE_WEIGHT_SIZE + PROJECT_WEIGHT_SIZE",
        "BIAS_SIZE = EXPAND_BIAS_SIZE + DEPTHWISE_BIAS_SIZE + PROJECT_BIAS_SIZE",
        "DECLARE_BRAM_WEIGHTS()",
        "#pragma HLS DATAFLOW",
    ):
        assert token in source
    assert "m_axi" not in source.lower()


def test_hls_export_is_strictly_bound_to_server_target_and_vlnv():
    script = (PILOT_HLS / "run_hls.tcl").read_text(encoding="utf-8")
    for token in (
        "xczu7ev-ffvc1156-2-e",
        "create_clock -period 5.000",
        "csynth_design",
        "export_design -format ip_catalog",
        "hwnas.org",
        "zcu104_stage4_k3_e3_kernel",
        "component.xml",
    ):
        assert token in script
    assert "xc7k70" not in script
    assert "2023.2" not in script


def test_pilot_bd_keeps_dma_shell_and_adds_only_parameter_brams():
    script = PILOT_BD.read_text(encoding="utf-8")
    for token in (
        "zynq_ultra_ps_e",
        "axi_dma",
        "operator_benchmark_wrapper:1.0",
        "zcu104_stage4_k3_e3_kernel:1.0",
        "weights_axi_bram_ctrl",
        "biases_axi_bram_ctrl",
        "weights_bram",
        "biases_bram",
        "0xA0000000",
        "0xA0010000",
        "0xA0100000",
        "0xA0200000",
        "validate_bd_design",
        "launch_runs synth_1",
        "launch_runs impl_1 -to_step route_design",
        "write_bitstream",
        "write_hw_platform -fixed -include_bit",
        "tns_ns",
    ):
        assert token in script
    assert "stage4_skip" not in script
    assert "2023.2" not in script


def test_pilot_status_excludes_runtime_latency_power_and_other_operators():
    status = json.loads(
        (
            ROOT
            / "artifacts"
            / "zcu104_dma_shell_mbconv_k3_e3_pilot_v1"
            / "status.json"
        ).read_text(encoding="utf-8")
    )
    assert status["scope"] == "one real compute operator only"
    assert status["operator_contract"]["weights_bytes"] == 7008
    assert status["operator_contract"]["biases_bytes"] == 896
    assert status["gates"]["board_runtime"] == "NOT_RUN"
    assert status["gates"]["latency"] == "NOT_MEASURED"
    assert status["gates"]["power"] == "NOT_MEASURED"


def test_pilot_software_header_freezes_parameter_windows_without_runtime_claim():
    header = (
        ROOT
        / "hardware"
        / "zcu104_dma_shell_v1"
        / "software"
        / "zcu104_dma_shell_mbconv_k3_e3_pilot.h"
    ).read_text(encoding="utf-8")
    for token in (
        "0x00040002",
        "0xA0100000",
        "0xA0200000",
        "7008u",
        "224u",
        "896u",
        "does not run that transaction",
    ):
        assert token in header
