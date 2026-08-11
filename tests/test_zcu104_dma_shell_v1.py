from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = ROOT / "hardware" / "zcu104_dma_shell_v1"


def test_platform_contract_freezes_dma_registers_and_latency_boundary():
    contract = json.loads(
        (PLATFORM_ROOT / "contracts" / "platform_contract.json").read_text(encoding="utf-8")
    )
    assert contract["branch"] == "codex/zcu104-dma-shell-v1"
    assert contract["target"]["part"] == "xczu7ev-ffvc1156-2-e"
    assert contract["transport"]["mm2s_axis_bits"] == 256
    assert contract["transport"]["s2mm_axis_bits"] == 256
    assert contract["transport"]["mm2s_memory_map_bits"] == 256
    assert contract["transport"]["s2mm_memory_map_bits"] == 256
    assert contract["address_map"]["operator_benchmark_wrapper"].startswith("0xA0000000")
    assert contract["address_map"]["axi_dma"].startswith("0xA0010000")
    registers = {entry["name"]: entry for entry in contract["registers"]}
    assert registers["CONTROL"]["offset"] == "0x00"
    assert registers["STATUS"]["offset"] == "0x04"
    assert registers["EXECUTION_COUNT_LO"]["offset"] == "0x08"
    assert registers["OPERATOR_CORE_LATENCY_CYCLES_LO"]["offset"] == "0x10"
    assert "DMA startup" in contract["counter_definition"]["operator_core_latency_cycles"]
    assert "Software timestamp" in contract["counter_definition"]["ps_dma_end_to_end_latency"]


def test_stage4_buffer_protocol_is_byte_exact_and_full_width():
    contract = json.loads(
        (PLATFORM_ROOT / "contracts" / "platform_contract.json").read_text(encoding="utf-8")
    )
    stage4 = contract["buffers"]["stage4"]
    assert stage4["bytes"] == 32 * 28 * 28
    assert stage4["dma_words_256b"] == 28 * 28
    assert stage4["bytes"] == stage4["dma_words_256b"] * 32
    assert stage4["dma_transfer_length_bytes"] == 25088
    assert stage4["tlast_word_index"] == 783


def test_seven_retained_operator_interfaces_are_audited():
    audit = json.loads(
        (ROOT / "artifacts" / "zcu104_dma_shell_v1" / "operator_interface_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "PASS_7_OF_7_EXISTING_OPERATOR_INTERFACES_AUDITED"
    assert len(audit["rows"]) == 7
    assert len(audit["blocked_not_audited"]) == 2
    for row in audit["rows"]:
        assert row["tensor_flatten_order"] == "pixel_major_HWC"
        assert row["input_axis"]["tkeep_bits"] == row["input_axis"]["tdata_bits"] // 8
        assert row["output_axis"]["tstrb_bits"] == row["output_axis"]["tdata_bits"] // 8
        if row["stage"] == "stage2":
            assert (row["input_axis"]["tdata_bits"], row["output_axis"]["tdata_bits"]) == (128, 192)
        else:
            assert (row["input_axis"]["tdata_bits"], row["output_axis"]["tdata_bits"]) == (256, 256)


def test_wrapper_is_axis_only_and_exposes_core_handshake_counter():
    rtl = (PLATFORM_ROOT / "rtl" / "operator_benchmark_wrapper.v").read_text(encoding="utf-8")
    lowered = rtl.lower()
    assert "m_axi_" not in lowered
    assert "s_axis_dma" in lowered
    assert "m_axis_dma" in lowered
    assert "m_axis_core" in lowered
    assert "s_axis_core" in lowered
    assert "core_latency_cycles" in rtl
    assert "input_handshake" in rtl
    assert "final_output_handshake" in rtl
    assert "error_flags[8]" in rtl


def test_skip_core_keeps_frozen_pixel_major_identity_semantics():
    source = (PLATFORM_ROOT / "hls" / "skip_core" / "zcu104_stage4_skip_core.cpp").read_text(
        encoding="utf-8"
    )
    assert "kChannels = 32" in source
    assert "kHeight = 28" in source
    assert "kWidth = 28" in source
    assert "out_packet.data = in_packet.data" in source
    assert "out_packet.keep = -1" in source
    assert "out_packet.strb = -1" in source
    assert "ap_ctrl_hs port=return" in source
    assert "m_axi" not in source.lower()


def test_skip_hls_uses_vitis_2022_2_compatible_relative_project_name():
    script = (PLATFORM_ROOT / "hls" / "skip_core" / "run_hls.tcl").read_text(
        encoding="utf-8"
    )
    assert "cd $result_root" in script
    assert "open_project -reset project" in script
    assert "open_project -reset $project_dir" not in script


def test_vivado_script_contains_complete_ps_dma_shell_and_target_only_implementation():
    script = (PLATFORM_ROOT / "vivado" / "build_bd.tcl").read_text(encoding="utf-8")
    for token in (
        "xczu7ev-ffvc1156-2-e",
        "zynq_ultra_ps_e",
        "axi_dma",
        "smartconnect",
        "proc_sys_reset",
        "M_AXIS_MM2S",
        "S_AXIS_S2MM",
        "S_AXI_HPC0_FPD",
        "M_AXI_HPM0_FPD",
        "0xA0000000",
        "0xA0010000",
        "launch_runs impl_1 -to_step write_bitstream",
        "write_hw_platform -fixed -include_bit",
    ):
        assert token in script
    assert "xc7k70" not in script


def test_pl_bd_preflight_integrates_dma_wrapper_and_real_exported_skip_ip():
    script = (PLATFORM_ROOT / "vivado" / "pl_bd_preflight.tcl").read_text(encoding="utf-8")
    for token in (
        "axi_dma",
        "operator_benchmark_wrapper:1.0",
        "zcu104_stage4_skip_core:1.0",
        "M_AXIS_MM2S",
        "S_AXIS_S2MM",
        "input_stream",
        "output_stream",
        "validate_bd_design",
    ):
        assert token in script
    assert "launch_runs" not in script
    assert "write_bitstream" not in script


def test_integration_sim_selects_actual_exported_hls_skip_rtl():
    script = (PLATFORM_ROOT / "rtl" / "run_wrapper_hls_core_sim.tcl").read_text(
        encoding="utf-8"
    )
    testbench = (PLATFORM_ROOT / "rtl" / "tb_operator_benchmark_wrapper.sv").read_text(
        encoding="utf-8"
    )
    assert "ACTUAL_HLS_CORE" in script
    assert "impl ip hdl verilog" in script
    assert "zcu104_stage4_skip_core actual_skip_core" in testbench
    wrapper = (PLATFORM_ROOT / "rtl" / "operator_benchmark_wrapper.v").read_text(
        encoding="utf-8"
    )
    assert "core_ap_start && core_ap_ready" in wrapper
