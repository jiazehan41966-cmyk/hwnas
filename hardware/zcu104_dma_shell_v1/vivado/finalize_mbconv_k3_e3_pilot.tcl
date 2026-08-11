set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_mbconv_k3_e3_pilot_v1 vivado]
set project_dir [file join $result_root project]
set project_file [file join $project_dir zcu104_dma_shell_mbconv_k3_e3_pilot_v1.xpr]
set part_name {xczu7ev-ffvc1156-2-e}
set board_part_name {xilinx.com:zcu104:part0:1.1}

if {![file exists $project_file]} {
    error "Missing Vivado project: $project_file"
}

open_project $project_file
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Synthesis is not complete: [get_property STATUS [get_runs synth_1]]"
}
if {[get_property STATUS [get_runs impl_1]] ne "route_design Complete!"} {
    error "Implementation is not complete: [get_property STATUS [get_runs impl_1]]"
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
set timing_met_json false
if {$wns >= 0.0 && $tns == 0.0} {
    set timing_met_json true
}

set timing_json [open [file join $result_root timing_gate.json] w]
puts $timing_json "{"
puts $timing_json "  \"schema_version\": 1,"
puts $timing_json "  \"wns_ns\": $wns,"
puts $timing_json "  \"tns_ns\": $tns,"
puts $timing_json "  \"timing_met\": $timing_met_json"
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
puts $result_json "  \"timing_met\": $timing_met_json,"
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
