# New H2 bring-up shell. Reuses PS/SmartConnect/DMA topology, direct HLS AXI-Lite.
set root [file dirname [file normalize [info script]]]
set part xczu7ev-ffvc1156-2-e
create_project h2_dma [file join $root vivado_project] -part $part
if {[llength [get_board_parts -quiet xilinx.com:zcu104:part0:*]] == 0} {error "ZCU104 board preset required"}
set_property board_part [lindex [get_board_parts xilinx.com:zcu104:part0:*] 0] [current_project]
set_property ip_repo_paths [list [file join $root hls_project solution1 impl ip]] [current_project]
update_ip_catalog
create_bd_design h2_dma_bd
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {200}] $ps
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* dma]
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_sg_length_width {26} CONFIG.c_include_mm2s_dre {1} CONFIG.c_include_s2mm_dre {1} CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_m_axi_s2mm_data_width {256} CONFIG.c_s_axis_s2mm_tdata_width {256}] $dma
set core [create_bd_cell -type ip -vlnv hwnas.org:hls:h2_dma:1.0 core]
set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $ctrl
set mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* mem]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] $mem
set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst]
set_property CONFIG.C_EXT_RESET_HIGH 0 $rst
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M00_AXI] [get_bd_intf_pins $core/s_axi_control]
connect_bd_intf_net [get_bd_intf_pins $ctrl/M01_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
foreach pair [list [list $dma/M_AXI_MM2S $mem/S00_AXI] [list $dma/M_AXI_S2MM $mem/S01_AXI] [list $core/m_axi_parameters $mem/S02_AXI] [list $mem/M00_AXI $ps/S_AXI_HPC0_FPD] [list $dma/M_AXIS_MM2S $core/input_stream] [list $core/output_stream $dma/S_AXIS_S2MM]] {connect_bd_intf_net [get_bd_intf_pins [lindex $pair 0]] [get_bd_intf_pins [lindex $pair 1]]}
foreach pin [list $ps/maxihpm0_fpd_aclk $ps/saxihpc0_fpd_aclk $ctrl/aclk $mem/aclk $dma/s_axi_lite_aclk $dma/m_axi_mm2s_aclk $dma/m_axi_s2mm_aclk $core/ap_clk $rst/slowest_sync_clk] {connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $pin]}
connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
foreach pin [list $ctrl/aresetn $mem/aresetn $dma/axi_resetn $core/ap_rst_n] {connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $pin]}
assign_bd_address
validate_bd_design
save_bd_design
set bd [get_files h2_dma_bd.bd]
generate_target all $bd
add_files -norecurse [make_wrapper -files $bd -top]
update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "implementation failed"}
open_run impl_1
report_timing_summary -file [file join $root timing.rpt]
report_utilization -hierarchical -file [file join $root utilization_hier.rpt]
report_route_status -file [file join $root route_status.rpt]
write_hw_platform -fixed -include_bit -force [file join $root h2_dma.xsa]
set paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $paths]!=1 || [get_property SLACK $paths]<0} {error "timing not closed"}
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $hold]!=1 || [get_property SLACK $hold]<0} {error "hold not closed"}
set f [open [file join $root ROUTE_TIMING_PASS] w]
puts $f "Tool [version -short]; scoped H2 DMA bring-up only; not full detector"
close $f
exit
