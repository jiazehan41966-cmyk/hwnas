set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_mbconv_k3_e3_pilot_v1 vivado]
set project_dir [file join $result_root project]
set wrapper_ip_root [file join $result_root ip_repo operator_benchmark_wrapper]
set hls_ip_repo [file join $repo_root results zcu104_dma_shell_mbconv_k3_e3_pilot_v1 hls project solution1 impl ip]
set wrapper_rtl [file join $repo_root hardware zcu104_dma_shell_v1 rtl operator_benchmark_wrapper.v]
set part_name {xczu7ev-ffvc1156-2-e}
set board_part_name {xilinx.com:zcu104:part0:1.1}

if {![file exists [file join $hls_ip_repo component.xml]]} {
    error "Missing exported HLS IP at $hls_ip_repo; run run_hls.tcl first."
}

file mkdir $result_root
file mkdir [file dirname $wrapper_ip_root]

# Package the custom wrapper so its AXI4-Lite address block is explicit in IP Integrator.
create_project -force wrapper_ip_pack [file join $result_root wrapper_ip_pack] -part $part_name
add_files $wrapper_rtl
set_property top operator_benchmark_wrapper [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $wrapper_ip_root -vendor hwnas.org -library user -taxonomy /HWNAS -import_files -set_current true
set wrapper_core [ipx::current_core]
set_property name operator_benchmark_wrapper $wrapper_core
set_property display_name {HWNAS Operator Benchmark Wrapper} $wrapper_core
set_property description {AXI DMA framing, control, status, and operator-core latency counter} $wrapper_core
set_property version 1.0 $wrapper_core
set_property core_revision 1 $wrapper_core

set ctrl_if [ipx::get_bus_interfaces S_AXI_CTRL -of_objects $wrapper_core]
if {[llength $ctrl_if] != 1} {
    error "S_AXI_CTRL was not inferred while packaging the wrapper."
}
set memory_maps [ipx::get_memory_maps -of_objects $wrapper_core]
if {[llength $memory_maps] != 1} {
    error "Expected exactly one inferred AXI-Lite memory map, found [llength $memory_maps]."
}
set memory_map [lindex $memory_maps 0]
set address_blocks [ipx::get_address_blocks -of_objects $memory_map]
if {[llength $address_blocks] != 1} {
    error "Expected exactly one inferred AXI-Lite address block, found [llength $address_blocks]."
}
set address_block [lindex $address_blocks 0]
set_property range 4096 $address_block
set_property width 32 $address_block
set_property usage register $address_block
set_property slave_memory_map_ref S_AXI_CTRL $ctrl_if

foreach register_spec {
    {CONTROL 0x00 read-write}
    {STATUS 0x04 read-only}
    {EXECUTION_COUNT_LO 0x08 read-only}
    {EXECUTION_COUNT_HI 0x0c read-only}
    {OPERATOR_CORE_LATENCY_CYCLES_LO 0x10 read-only}
    {OPERATOR_CORE_LATENCY_CYCLES_HI 0x14 read-only}
    {EXPECTED_INPUT_WORDS 0x18 read-only}
    {EXPECTED_OUTPUT_WORDS 0x1c read-only}
    {INPUT_WORDS_SEEN 0x20 read-only}
    {OUTPUT_WORDS_SEEN 0x24 read-only}
    {ERROR_FLAGS 0x28 read-only}
    {SHELL_ID 0x2c read-only}
    {VERSION 0x30 read-only}
    {OPERATOR_ID 0x34 read-only}
    {AXIS_DATA_BITS 0x38 read-only}
    {CLOCK_HZ 0x3c read-only}
} {
    lassign $register_spec reg_name reg_offset reg_access
    set reg_obj [ipx::add_register $reg_name $address_block]
    set_property address_offset $reg_offset $reg_obj
    set_property size 32 $reg_obj
    set_property access $reg_access $reg_obj
}

foreach busif {S_AXI_CTRL S_AXIS_DMA M_AXIS_DMA M_AXIS_CORE S_AXIS_CORE} {
    ipx::associate_bus_interfaces -busif $busif -clock aclk $wrapper_core
}
ipx::create_xgui_files $wrapper_core
ipx::update_checksums $wrapper_core
ipx::check_integrity -quiet $wrapper_core
ipx::save_core $wrapper_core
close_project

set board_repo_candidates {}
if {[info exists ::env(ZCU104_BOARD_REPO)]} {
    lappend board_repo_candidates $::env(ZCU104_BOARD_REPO)
}
lappend board_repo_candidates {/opt/Xilinx/Vivado/2022.2/data/boards/board_files}
foreach board_repo $board_repo_candidates {
    if {[file isdirectory $board_repo]} {
        set_param board.repoPaths [list $board_repo]
        break
    }
}

create_project -force zcu104_dma_shell_mbconv_k3_e3_pilot_v1 $project_dir -part $part_name
set board_parts [get_board_parts -quiet $board_part_name]
if {[llength $board_parts] == 1} {
    set_property board_part $board_part_name [current_project]
} else {
    puts "WARNING: ZCU104 board part not registered; using explicit part and PS settings."
}

set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property ip_repo_paths [list $wrapper_ip_root $hls_ip_repo] [current_project]
update_ip_catalog

create_bd_design dma_shell_mbconv_k3_e3_bd

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* zynq_ultra_ps_e_0]
if {[llength $board_parts] == 1} {
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
        -config {apply_board_preset "1"} $ps
}
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} \
    CONFIG.PSU__USE__IRQ0 {0} \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200} \
] $ps

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

set ctrl_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {4}] $ctrl_smc
set mem_smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_smc_mem]
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] $mem_smc
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* proc_sys_reset_0]

set weights_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:* weights_axi_bram_ctrl]
set_property -dict [list CONFIG.DATA_WIDTH {32} CONFIG.SINGLE_PORT_BRAM {1}] $weights_ctrl
set weights_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:* weights_bram]
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.use_bram_block {Stand_Alone} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Depth_A {2048} \
    CONFIG.Write_Width_B {8} \
    CONFIG.Read_Width_B {8} \
    CONFIG.Enable_B {Use_ENB_Pin} \
] $weights_mem

set biases_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:* biases_axi_bram_ctrl]
set_property -dict [list CONFIG.DATA_WIDTH {32} CONFIG.SINGLE_PORT_BRAM {1}] $biases_ctrl
set biases_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:blk_mem_gen:* biases_bram]
set_property -dict [list \
    CONFIG.Memory_Type {True_Dual_Port_RAM} \
    CONFIG.use_bram_block {Stand_Alone} \
    CONFIG.Use_Byte_Write_Enable {true} \
    CONFIG.Byte_Size {8} \
    CONFIG.Write_Width_A {32} \
    CONFIG.Read_Width_A {32} \
    CONFIG.Write_Depth_A {256} \
    CONFIG.Write_Width_B {32} \
    CONFIG.Read_Width_B {32} \
    CONFIG.Enable_B {Use_ENB_Pin} \
] $biases_mem

set wrapper [create_bd_cell -type ip -vlnv hwnas.org:user:operator_benchmark_wrapper:1.0 operator_benchmark_wrapper_0]
set_property -dict [list \
    CONFIG.AXIS_DATA_BITS {256} \
    CONFIG.INPUT_WORDS {784} \
    CONFIG.OUTPUT_WORDS {784} \
    CONFIG.OPERATOR_ID {0x00040002} \
    CONFIG.CLOCK_HZ {200000000} \
] $wrapper
set core [create_bd_cell -type ip -vlnv hwnas.org:hls:zcu104_stage4_k3_e3_kernel:1.0 zcu104_stage4_k3_e3_kernel_0]

connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $ctrl_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M00_AXI] [get_bd_intf_pins $wrapper/S_AXI_CTRL]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M01_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M02_AXI] [get_bd_intf_pins $weights_ctrl/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl_smc/M03_AXI] [get_bd_intf_pins $biases_ctrl/S_AXI]

connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $mem_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $mem_smc/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $mem_smc/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0_FPD]

connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $wrapper/S_AXIS_DMA]
connect_bd_intf_net [get_bd_intf_pins $wrapper/M_AXIS_DMA] [get_bd_intf_pins $dma/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins $wrapper/M_AXIS_CORE] [get_bd_intf_pins $core/input_stream]
connect_bd_intf_net [get_bd_intf_pins $core/output_stream] [get_bd_intf_pins $wrapper/S_AXIS_CORE]
connect_bd_intf_net [get_bd_intf_pins $weights_ctrl/BRAM_PORTA] [get_bd_intf_pins $weights_mem/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins $core/weights_PORTA] [get_bd_intf_pins $weights_mem/BRAM_PORTB]
connect_bd_intf_net [get_bd_intf_pins $biases_ctrl/BRAM_PORTA] [get_bd_intf_pins $biases_mem/BRAM_PORTA]
connect_bd_intf_net [get_bd_intf_pins $core/biases_PORTA] [get_bd_intf_pins $biases_mem/BRAM_PORTB]

connect_bd_net [get_bd_pins $wrapper/core_ap_start] [get_bd_pins $core/ap_start]
connect_bd_net [get_bd_pins $core/ap_done] [get_bd_pins $wrapper/core_ap_done]
connect_bd_net [get_bd_pins $core/ap_idle] [get_bd_pins $wrapper/core_ap_idle]
connect_bd_net [get_bd_pins $core/ap_ready] [get_bd_pins $wrapper/core_ap_ready]

set pl_clk [get_bd_pins $ps/pl_clk0]
foreach clock_pin [list \
    [get_bd_pins $ps/maxihpm0_fpd_aclk] \
    [get_bd_pins $ps/saxihp0_fpd_aclk] \
    [get_bd_pins $ctrl_smc/aclk] \
    [get_bd_pins $mem_smc/aclk] \
    [get_bd_pins $dma/s_axi_lite_aclk] \
    [get_bd_pins $dma/m_axi_mm2s_aclk] \
    [get_bd_pins $dma/m_axi_s2mm_aclk] \
    [get_bd_pins $weights_ctrl/s_axi_aclk] \
    [get_bd_pins $biases_ctrl/s_axi_aclk] \
    [get_bd_pins $rst/slowest_sync_clk] \
    [get_bd_pins $wrapper/aclk] \
    [get_bd_pins $core/ap_clk] \
] {
    connect_bd_net $pl_clk $clock_pin
}
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]

set peripheral_resetn [get_bd_pins $rst/peripheral_aresetn]
foreach reset_pin [list \
    [get_bd_pins $ctrl_smc/aresetn] \
    [get_bd_pins $mem_smc/aresetn] \
    [get_bd_pins $dma/axi_resetn] \
    [get_bd_pins $weights_ctrl/s_axi_aresetn] \
    [get_bd_pins $biases_ctrl/s_axi_aresetn] \
    [get_bd_pins $wrapper/aresetn] \
    [get_bd_pins $core/ap_rst_n] \
] {
    connect_bd_net $peripheral_resetn $reset_pin
}

# Polling is the frozen v1 PS-control method. Interrupt outputs remain optional
# and intentionally unconnected until the software runtime is selected.
assign_bd_address

set ps_data_space [get_bd_addr_spaces $ps/Data]
set wrapper_slave_seg [get_bd_addr_segs $wrapper/S_AXI_CTRL/reg0]
set dma_slave_seg [get_bd_addr_segs $dma/S_AXI_LITE/Reg]
set weights_slave_seg [get_bd_addr_segs $weights_ctrl/S_AXI/Mem0]
set biases_slave_seg [get_bd_addr_segs $biases_ctrl/S_AXI/Mem0]
assign_bd_address -offset 0xA0000000 -range 64K -target_address_space $ps_data_space $wrapper_slave_seg -force
assign_bd_address -offset 0xA0010000 -range 64K -target_address_space $ps_data_space $dma_slave_seg -force
assign_bd_address -offset 0xA0100000 -range 64K -target_address_space $ps_data_space $weights_slave_seg -force
assign_bd_address -offset 0xA0200000 -range 64K -target_address_space $ps_data_space $biases_slave_seg -force

save_bd_design
validate_bd_design
write_bd_tcl -force [file join $result_root dma_shell_mbconv_k3_e3_bd.tcl]

set bd_file [get_files dma_shell_mbconv_k3_e3_bd.bd]
generate_target all $bd_file
set wrapper_files [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_files
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
}

launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property STATUS [get_runs impl_1]] ne "route_design Complete!"} {
    error "Implementation failed: [get_property STATUS [get_runs impl_1]]"
}

open_run impl_1
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose \
    -max_paths 20 -input_pins -file [file join $result_root post_route_timing_summary.rpt]
report_utilization -hierarchical -file [file join $result_root post_route_utilization_hier.rpt]
report_utilization -file [file join $result_root post_route_utilization.rpt]
report_clock_utilization -file [file join $result_root post_route_clock_utilization.rpt]
write_checkpoint -force [file join $result_root zcu104_dma_shell_mbconv_k3_e3_pilot_v1_routed.dcp]

set worst_path [get_timing_paths -delay_type max -max_paths 1]
set wns [get_property SLACK $worst_path]
set tns 0.0
foreach failing_path [get_timing_paths -delay_type max -max_paths 100000 -nworst 1 -slack_lesser_than 0.0] {
    set tns [expr {$tns + [get_property SLACK $failing_path]}]
}

set timing_json [open [file join $result_root timing_gate.json] w]
puts $timing_json "{"
puts $timing_json "  \"schema_version\": 1,"
puts $timing_json "  \"wns_ns\": $wns,"
puts $timing_json "  \"tns_ns\": $tns,"
puts $timing_json "  \"timing_met\": [expr {$wns >= 0.0 && $tns == 0.0 ? \"true\" : \"false\"}]"
puts $timing_json "}"
close $timing_json

if {$wns < 0.0 || $tns != 0.0} {
    error "Post-route timing gate failed: WNS=$wns ns, TNS=$tns ns. Bitstream and XSA intentionally not generated."
}

write_bitstream -force [file join $result_root zcu104_dma_shell_mbconv_k3_e3_pilot_v1.bit]
write_hw_platform -fixed -include_bit -force [file join $result_root zcu104_dma_shell_mbconv_k3_e3_pilot_v1.xsa]

set result_json [open [file join $result_root build_result.json] w]
puts $result_json "{"
puts $result_json "  \"schema_version\": 1,"
puts $result_json "  \"status\": \"PASS_MBConv_K3_E3_DMA_SHELL_FULL_IMPLEMENTATION\","
puts $result_json "  \"tool\": \"Vivado 2022.2\","
puts $result_json "  \"target_part\": \"$part_name\","
puts $result_json "  \"board_part\": \"$board_part_name\","
puts $result_json "  \"clock_hz\": 200000000,"
puts $result_json "  \"wns_ns\": $wns,"
puts $result_json "  \"tns_ns\": $tns,"
puts $result_json "  \"timing_met\": [expr {$wns >= 0.0 ? \"true\" : \"false\"}],"
puts $result_json "  \"wrapper_base\": \"0xA0000000\","
puts $result_json "  \"dma_base\": \"0xA0010000\","
puts $result_json "  \"weights_base\": \"0xA0100000\","
puts $result_json "  \"biases_base\": \"0xA0200000\","
puts $result_json "  \"operator\": \"stage4_mbconv_k3_e3\","
puts $result_json "  \"operator_id\": \"0x00040002\","
puts $result_json "  \"hls_vlnv\": \"hwnas.org:hls:zcu104_stage4_k3_e3_kernel:1.0\","
puts $result_json "  \"board_programmed\": false,"
puts $result_json "  \"ps_runtime_executed\": false,"
puts $result_json "  \"board_latency_measured\": false,"
puts $result_json "  \"power_measured\": false,"
puts $result_json "  \"claim_boundary\": \"Full Vivado implementation and bitstream generation only; no board execution or measurement.\""
puts $result_json "}"
close $result_json

close_project
exit
