#!/usr/bin/env python3
"""Generate the ZCU104 PS-DDR/AXI-DMA block-design and export Tcl."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_PART = "xczu7ev-ffvc1156-2-e"
BOARD_PART = "xilinx.com:zcu104:part0:1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip-repo", required=True, help="Vitis HLS exported IP repository")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--kernel-vlnv",
        default="xilinx.com:hls:fourstage_top_hls:1.0",
    )
    parser.add_argument("--project-name", default="zcu104_fourstage")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--validate-ip-repo",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def tcl_quote(path: Path) -> str:
    return "{" + path.as_posix() + "}"


def build_tcl(
    *,
    ip_repo: Path,
    output_dir: Path,
    kernel_vlnv: str,
    project_name: str,
    jobs: int,
) -> str:
    project_dir = output_dir / "vivado_project"
    bitstream_dir = output_dir / "bitstream"
    reports_dir = output_dir / "reports"
    export_dir = output_dir / "platform"
    return f"""set project_name {{{project_name}}}
set target_part {{{TARGET_PART}}}
set board_part {{{BOARD_PART}}}
set ip_repo {tcl_quote(ip_repo)}
set project_dir {tcl_quote(project_dir)}
set bitstream_dir {tcl_quote(bitstream_dir)}
set reports_dir {tcl_quote(reports_dir)}
set export_dir {tcl_quote(export_dir)}

create_project $project_name $project_dir -part $target_part -force
set_property board_part $board_part [current_project]
set_property ip_repo_paths [list $ip_repo] [current_project]
update_ip_catalog

if {{[llength [get_ipdefs -all {{{kernel_vlnv}}}]] == 0}} {{
  error "Kernel IP not found in repository: {kernel_vlnv}"
}}

create_bd_design "zcu104_system"
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps_0]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {{apply_board_preset "1"}} $ps
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {{1}} \
  CONFIG.PSU__USE__S_AXI_GP0 {{1}} \
  CONFIG.PSU__USE__IRQ0 {{1}} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {{200}} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_0]
set kernel [create_bd_cell -type ip -vlnv {kernel_vlnv} fourstage_0]
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list \
  CONFIG.c_include_sg {{0}} \
  CONFIG.c_sg_length_width {{26}} \
  CONFIG.c_m_axi_mm2s_data_width {{64}} \
  CONFIG.c_m_axi_s2mm_data_width {{64}} \
  CONFIG.c_m_axis_mm2s_tdata_width {{8}} \
  CONFIG.c_s_axis_s2mm_tdata_width {{8}} \
] $dma

set mem_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 mem_smc]
set_property -dict [list CONFIG.NUM_SI {{6}} CONFIG.NUM_MI {{1}}] $mem_smc
set ctrl_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 ctrl_smc]
set_property -dict [list CONFIG.NUM_SI {{1}} CONFIG.NUM_MI {{2}}] $ctrl_smc
set irq_concat [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 irq_concat]
set_property CONFIG.NUM_PORTS {{3}} $irq_concat

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $ctrl_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M00_AXI] [get_bd_intf_pins $kernel/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M01_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]

connect_bd_intf_net [get_bd_intf_pins $kernel/m_axi_gmem0] [get_bd_intf_pins $mem_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $kernel/m_axi_gmem1] [get_bd_intf_pins $mem_smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $kernel/m_axi_gmem2] [get_bd_intf_pins $mem_smc/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins $kernel/m_axi_gmem3] [get_bd_intf_pins $mem_smc/S03_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $mem_smc/S04_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $mem_smc/S05_AXI]
connect_bd_intf_net [get_bd_intf_pins $mem_smc/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]

connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $kernel/input_stream]
connect_bd_intf_net [get_bd_intf_pins $kernel/output_stream] [get_bd_intf_pins $dma/S_AXIS_S2MM]

set pl_clk [get_bd_pins $ps/pl_clk0]
connect_bd_net $pl_clk [get_bd_pins $rst/slowest_sync_clk]
connect_bd_net $pl_clk [get_bd_pins $kernel/ap_clk]
connect_bd_net $pl_clk [get_bd_pins $dma/s_axi_lite_aclk]
connect_bd_net $pl_clk [get_bd_pins $dma/m_axi_mm2s_aclk]
connect_bd_net $pl_clk [get_bd_pins $dma/m_axi_s2mm_aclk]
connect_bd_net $pl_clk [get_bd_pins $dma/m_axis_mm2s_aclk]
connect_bd_net $pl_clk [get_bd_pins $dma/s_axis_s2mm_aclk]
connect_bd_net $pl_clk [get_bd_pins $mem_smc/aclk]
connect_bd_net $pl_clk [get_bd_pins $ctrl_smc/aclk]
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]

set pl_resetn [get_bd_pins $rst/peripheral_aresetn]
connect_bd_net $pl_resetn [get_bd_pins $kernel/ap_rst_n]
connect_bd_net $pl_resetn [get_bd_pins $dma/axi_resetn]
connect_bd_net $pl_resetn [get_bd_pins $mem_smc/aresetn]
connect_bd_net $pl_resetn [get_bd_pins $ctrl_smc/aresetn]

connect_bd_net [get_bd_pins $kernel/interrupt] [get_bd_pins $irq_concat/In0]
connect_bd_net [get_bd_pins $dma/mm2s_introut] [get_bd_pins $irq_concat/In1]
connect_bd_net [get_bd_pins $dma/s2mm_introut] [get_bd_pins $irq_concat/In2]
connect_bd_net [get_bd_pins $irq_concat/dout] [get_bd_pins $ps/pl_ps_irq0]

assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files zcu104_system.bd]
make_wrapper -files [get_files zcu104_system.bd] -top
add_files -norecurse [glob $project_dir/$project_name.gen/sources_1/bd/zcu104_system/hdl/zcu104_system_wrapper.v]
set_property top zcu104_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs {int(jobs)}
wait_on_run synth_1
if {{[get_property STATUS [get_runs synth_1]] !~ "synth_design Complete*"}} {{
  error "ZCU104 synthesis failed: [get_property STATUS [get_runs synth_1]]"
}}
launch_runs impl_1 -to_step write_bitstream -jobs {int(jobs)}
wait_on_run impl_1
if {{[get_property STATUS [get_runs impl_1]] !~ "write_bitstream Complete*"}} {{
  error "ZCU104 implementation/bitstream failed: [get_property STATUS [get_runs impl_1]]"
}}

file mkdir $bitstream_dir
file mkdir $reports_dir
file mkdir $export_dir
open_run impl_1
report_timing_summary -file $reports_dir/timing_summary.rpt
report_utilization -hierarchical -file $reports_dir/utilization_hierarchical.rpt
report_power -file $reports_dir/vivado_power_estimate.rpt
write_bitstream -force $bitstream_dir/zcu104_fourstage.bit
write_hw_platform -fixed -include_bit -force -file $export_dir/zcu104_fourstage.xsa

set hwh_candidates [glob -nocomplain $project_dir/$project_name.gen/sources_1/bd/zcu104_system/hw_handoff/*.hwh]
if {{[llength $hwh_candidates] != 1}} {{
  error "Expected one HWH export, found [llength $hwh_candidates]"
}}
file copy -force [lindex $hwh_candidates 0] $export_dir/zcu104_fourstage.hwh
puts "ZCU104_PLATFORM_EXPORT_PASS"
exit
"""


def main() -> int:
    args = parse_args()
    ip_repo = Path(args.ip_repo).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.validate_ip_repo:
        components = list(ip_repo.rglob("component.xml")) if ip_repo.is_dir() else []
        if not components:
            raise FileNotFoundError(f"HLS IP component.xml not found under {ip_repo}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tcl_path = output_dir / "build_zcu104_platform.tcl"
    tcl_path.write_text(
        build_tcl(
            ip_repo=ip_repo,
            output_dir=output_dir,
            kernel_vlnv=str(args.kernel_vlnv),
            project_name=str(args.project_name),
            jobs=int(args.jobs),
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "campaign_id": "zcu104_full_campaign_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "GENERATED_NOT_RUN",
        "target_part": TARGET_PART,
        "board_part": BOARD_PART,
        "kernel_vlnv": str(args.kernel_vlnv),
        "ip_repo": str(ip_repo),
        "transport": {"type": "axi_dma", "id": "axi_dma_0"},
        "memory": "ps_ddr4_via_s_axi_hpc0_fpd",
        "clock": "ps_fclk0_200mhz",
        "interrupts": ["kernel", "dma_mm2s", "dma_s2mm"],
        "exports": ["bitstream", "xsa", "hwh", "timing", "utilization", "power_estimate"],
        "tcl": str(tcl_path),
        "claim_boundary": "Generated Tcl is not HLS, route, bitstream, XSA/HWH, board, latency, or power evidence.",
    }
    manifest_path = output_dir / "platform_generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
