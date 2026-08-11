set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 vivado pl_bd_preflight]
set project_dir [file join $result_root project]
set wrapper_ip_repo [file join $repo_root results zcu104_dma_shell_v1 vivado wrapper_package_preflight ip]
set hls_ip_repo [file join $repo_root results zcu104_dma_shell_v1 hls skip_core_portable_preflight project solution1 impl ip]

# This preflight is intentionally device independent.  It checks only that the
# packaged DMA shell, the actual exported Skip HLS IP, and AXI DMA form a valid
# IP-Integrator graph.  It is never used for synthesis, implementation, or a
# ZCU104 claim.
if {![file exists [file join $wrapper_ip_repo component.xml]]} {
    error "Missing wrapper packaging preflight at $wrapper_ip_repo."
}
if {![file exists [file join $hls_ip_repo component.xml]]} {
    error "Missing portable Skip HLS preflight at $hls_ip_repo."
}

set installed_parts [get_parts -quiet]
if {[llength $installed_parts] == 0} {
    error "No Vivado device part is installed; cannot run device-independent BD validation."
}
set preflight_part [lindex $installed_parts 0]

file mkdir $result_root
create_project -force pl_bd_preflight $project_dir -part $preflight_part
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $wrapper_ip_repo $hls_ip_repo] [current_project]
update_ip_catalog

create_bd_design pl_dma_core_preflight

set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_0]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_sg_length_width {26} \
    CONFIG.c_include_mm2s_dre {1} \
    CONFIG.c_include_s2mm_dre {1} \
    CONFIG.c_m_axi_mm2s_data_width {256} \
    CONFIG.c_m_axis_mm2s_tdata_width {256} \
    CONFIG.c_m_axi_s2mm_data_width {256} \
    CONFIG.c_s_axis_s2mm_tdata_width {256} \
    CONFIG.c_mm2s_burst_size {64} \
    CONFIG.c_s2mm_burst_size {64} \
] $dma

set wrapper [create_bd_cell -type ip -vlnv hwnas.org:user:operator_benchmark_wrapper:1.0 operator_benchmark_wrapper_0]
set_property -dict [list \
    CONFIG.AXIS_DATA_BITS {256} \
    CONFIG.INPUT_WORDS {784} \
    CONFIG.OUTPUT_WORDS {784} \
    CONFIG.OPERATOR_ID {0x00040001} \
    CONFIG.CLOCK_HZ {200000000} \
] $wrapper
set core [create_bd_cell -type ip -vlnv hwnas.org:preflight:zcu104_stage4_skip_core:1.0 zcu104_stage4_skip_core_0]

connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $wrapper/S_AXIS_DMA]
connect_bd_intf_net [get_bd_intf_pins $wrapper/M_AXIS_DMA] [get_bd_intf_pins $dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins $wrapper/M_AXIS_CORE] [get_bd_intf_pins $core/input_stream]
connect_bd_intf_net [get_bd_intf_pins $core/output_stream] [get_bd_intf_pins $wrapper/S_AXIS_CORE]

connect_bd_net [get_bd_pins $wrapper/core_ap_start] [get_bd_pins $core/ap_start]
connect_bd_net [get_bd_pins $core/ap_done] [get_bd_pins $wrapper/core_ap_done]
connect_bd_net [get_bd_pins $core/ap_idle] [get_bd_pins $wrapper/core_ap_idle]
connect_bd_net [get_bd_pins $core/ap_ready] [get_bd_pins $wrapper/core_ap_ready]

set clk [create_bd_port -dir I -type clk -freq_hz 200000000 aclk]
set resetn [create_bd_port -dir I -type rst aresetn]
set_property CONFIG.POLARITY ACTIVE_LOW $resetn

foreach clock_pin [list \
    [get_bd_pins $dma/s_axi_lite_aclk] \
    [get_bd_pins $dma/m_axi_mm2s_aclk] \
    [get_bd_pins $dma/m_axi_s2mm_aclk] \
    [get_bd_pins $wrapper/aclk] \
    [get_bd_pins $core/ap_clk] \
] {
    connect_bd_net $clk $clock_pin
}
foreach reset_pin [list \
    [get_bd_pins $dma/axi_resetn] \
    [get_bd_pins $wrapper/aresetn] \
    [get_bd_pins $core/ap_rst_n] \
] {
    connect_bd_net $resetn $reset_pin
}

foreach intf_pin [list \
    [get_bd_intf_pins $dma/S_AXI_LITE] \
    [get_bd_intf_pins $dma/M_AXI_MM2S] \
    [get_bd_intf_pins $dma/M_AXI_S2MM] \
    [get_bd_intf_pins $wrapper/S_AXI_CTRL] \
] {
    make_bd_intf_pins_external $intf_pin
}

assign_bd_address
save_bd_design
validate_bd_design
write_bd_tcl -force [file join $result_root pl_dma_core_preflight_bd.tcl]

set result_json [open [file join $result_root result.json] w]
puts $result_json "{"
puts $result_json "  \"schema_version\": 1,"
puts $result_json "  \"status\": \"PASS_DEVICE_INDEPENDENT_BD_CONNECTION_VALIDATION\","
puts $result_json "  \"preflight_part\": \"$preflight_part\","
puts $result_json {  "integrated_blocks": ["axi_dma", "operator_benchmark_wrapper", "stage4_skip_hls_ip"],}
puts $result_json "  \"axis_bits\": 256,"
puts $result_json "  \"claim_boundary\": \"IP Integrator interface validation only; no Zynq PS, ZCU104 target synthesis, implementation, bitstream, or board result.\""
puts $result_json "}"
close $result_json

close_project
exit
