# Invoke: vivado -mode batch -source build_shell.tcl -tclargs CASE_DIRECTORY
# Each case must already contain hls_project/solution1/impl/ip.
# This script never opens a hardware target or programs a board.
set source_root [file dirname [file normalize [info script]]]
if {$argc != 1} {error "provide exactly one generated case directory"}
set case_root [file normalize [lindex $argv 0]]
set output_root [file join $case_root vivado_shell]
set ip_root [file join $case_root hls_project solution1 impl ip]
if {![file isdirectory $ip_root]} {error "exported HLS IP is missing: $ip_root"}
if {[file exists $output_root]} {error "refusing to overwrite existing shell evidence: $output_root"}
file mkdir $output_root
set stages_path [file join $output_root shell_stages.jsonl]

proc read_text {path} {
    set fd [open $path r]
    set value [read $fd]
    close $fd
    return $value
}
proc json_quote {value} {
    return "\"[string map [list \\ \\\\ \" \\\" \n \\n \r \\r \t \\t] $value]\""
}
proc timed {name body} {
    global stages_path
    set started [clock milliseconds]
    set code [catch {uplevel 1 $body} value options]
    set ended [clock milliseconds]
    set fd [open $stages_path a]
    puts $fd [format {{"stage":%s,"start_epoch_ms":%d,"end_epoch_ms":%d,"wall_seconds":%.3f,"returncode":%d,"status":%s}} \
        [json_quote $name] $started $ended [expr {($ended-$started)/1000.0}] $code [json_quote [expr {$code ? "FAIL" : "PASS"}]]]
    close $fd
    if {$code} {return -options $options $value}
    return $value
}
proc one {objects description} {
    if {[llength $objects] != 1} {error "expected one $description, found [llength $objects]"}
    return [lindex $objects 0]
}
proc run_complete {name} {
    set run [get_runs $name]
    if {[get_property PROGRESS $run] ne "100%" || [regexp -nocase {error|fail} [get_property STATUS $run]]} {
        error "$name failed: [get_property STATUS $run]"
    }
}

timed shell_generation {
    create_project pilot_shell [file join $output_root project] -part xczu7ev-ffvc1156-2-e
    set presets [get_board_parts -quiet xilinx.com:zcu104:part0:*]
    if {![llength $presets]} {error "ZCU104 board preset required"}
    set board_part [lindex [lsort -dictionary $presets] end]
    set_property board_part $board_part [current_project]
    set_property ip_repo_paths [list $ip_root] [current_project]
    update_ip_catalog
    add_files -norecurse [file join $source_root phase_timer.v]
    update_compile_order -fileset sources_1
    create_bd_design pilot_bd
    set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps]
    apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
    set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} \
        CONFIG.PSU__USE__S_AXI_GP0 {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {187.5}] $ps
    set core [create_bd_cell -type ip -vlnv hwnas.org:hls:pilot_kernel:1.0 core]
    set ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl]
    set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {4}] $ctrl
    set mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* mem]
    # HLS may remove parameters for nearest2x; discover only expected bundles.
    set memory_bundles {}
    foreach bundle {inputs parameters outputs} {
        if {[llength [get_bd_intf_pins -quiet $core/m_axi_$bundle]] == 1} {lappend memory_bundles $bundle}
    }
    if {"inputs" ni $memory_bundles || "outputs" ni $memory_bundles} {error "required HLS input/output AXI master absent"}
    set_property -dict [list CONFIG.NUM_SI [llength $memory_bundles] CONFIG.NUM_MI {1}] $mem
    set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* rst]
    # C_EXT_RESET_HIGH is read-only; the active-low PS reset connection below
    # propagates its value. Check the generated XCI/HWH rather than forcing it.
    set high [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* reset_high]
    set low [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* reset_low]
    set_property CONFIG.CONST_VAL {1} $high
    set_property CONFIG.CONST_VAL {0} $low
    set timer [create_bd_cell -type module -reference phase_timer timer]
    set gpio_names [list timer_counts timer_sequence timer_identity]
    foreach name $gpio_names {
        set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:* $name]
        set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_INPUTS {1} CONFIG.C_ALL_INPUTS_2 {1} \
            CONFIG.C_GPIO_WIDTH {32} CONFIG.C_GPIO2_WIDTH {32}] $gpio
    }
    connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_HPM0_FPD] [get_bd_intf_pins $ctrl/S00_AXI]
    connect_bd_intf_net [get_bd_intf_pins $ctrl/M00_AXI] [get_bd_intf_pins $core/s_axi_control]
    set index 1
    foreach gpio $gpio_names {
        connect_bd_intf_net [get_bd_intf_pins $ctrl/[format M%02d_AXI $index]] [get_bd_intf_pins $gpio/S_AXI]
        incr index
    }
    set index 0
    foreach bundle $memory_bundles {
        connect_bd_intf_net [get_bd_intf_pins $core/m_axi_$bundle] [get_bd_intf_pins $mem/[format S%02d_AXI $index]]
        incr index
    }
    connect_bd_intf_net [get_bd_intf_pins $mem/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HPC0_FPD]
    set phase_candidates {}
    foreach name {phase phase_V} {
        set candidate [get_bd_pins -quiet $core/$name]
        if {[llength $candidate]} {lappend phase_candidates $candidate}
    }
    set phase_pin [one $phase_candidates "HLS phase or phase_V output"]
    if {[get_property DIR $phase_pin] ne "O" || [get_property LEFT $phase_pin] ne "7" || [get_property RIGHT $phase_pin] ne "0"} {
        error "HLS phase signal must be an 8-bit output"
    }
    set phase_valid_candidates {}
    foreach name {phase_ap_vld phase_V_ap_vld} {
        set candidate [get_bd_pins -quiet $core/$name]
        if {[llength $candidate]} {lappend phase_valid_candidates $candidate}
    }
    set phase_valid_pin [one $phase_valid_candidates "HLS phase ap_vld strobe"]
    if {[get_property DIR $phase_valid_pin] ne "O"} {error "phase ap_vld must be an output"}
    connect_bd_net $phase_pin [get_bd_pins $timer/phase]
    connect_bd_net $phase_valid_pin [get_bd_pins $timer/phase_valid]
    foreach pair {{compute_cycles timer_counts/gpio_io_i} {transfer_cycles timer_counts/gpio2_io_i} \
                  {total_cycles timer_sequence/gpio_io_i} {sequence_count timer_sequence/gpio2_io_i} \
                  {identity timer_identity/gpio_io_i} {status timer_identity/gpio2_io_i}} {
        connect_bd_net [get_bd_pins $timer/[lindex $pair 0]] [get_bd_pins [lindex $pair 1]]
    }
    set clock_pins [list $ps/maxihpm0_fpd_aclk $ps/saxihpc0_fpd_aclk $ctrl/aclk $mem/aclk $core/ap_clk $rst/slowest_sync_clk $timer/clk]
    set reset_pins [list $ctrl/aresetn $mem/aresetn $core/ap_rst_n $timer/aresetn]
    foreach gpio $gpio_names {
        lappend clock_pins $gpio/s_axi_aclk
        lappend reset_pins $gpio/s_axi_aresetn
    }
    foreach pin $clock_pins {connect_bd_net [get_bd_pins $ps/pl_clk0] [get_bd_pins $pin]}
    connect_bd_net [get_bd_pins $ps/pl_resetn0] [get_bd_pins $rst/ext_reset_in]
    # The preset propagates both external and auxiliary resets as active-low.
    # Their inactive tieoff is 1. Debug reset remains active-high/inactive 0.
    connect_bd_net [get_bd_pins $high/dout] [get_bd_pins $rst/dcm_locked] [get_bd_pins $rst/aux_reset_in]
    connect_bd_net [get_bd_pins $low/dout] [get_bd_pins $rst/mb_debug_sys_rst]
    foreach pin $reset_pins {connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins $pin]}
    assign_bd_address
    foreach pair {{core/s_axi_control 0xA0000000} {timer_counts/S_AXI 0xA0010000} \
                  {timer_sequence/S_AXI 0xA0020000} {timer_identity/S_AXI 0xA0030000}} {
        set target [one [get_bd_addr_segs -of_objects [get_bd_intf_pins [lindex $pair 0]]] "AXI register segment"]
        assign_bd_address -force -offset [lindex $pair 1] -range 0x00010000 \
            -target_address_space [get_bd_addr_spaces $ps/Data] $target
    }
    validate_bd_design
    foreach {parameter expected} {C_EXT_RESET_HIGH 0 C_AUX_RESET_HIGH 0} {
        if {[get_property CONFIG.$parameter $rst] ne $expected} {
            error "Unexpected reset polarity $parameter; inactive auxiliary tieoff would be invalid"
        }
    }
    set actual_pl0_mhz [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
    if {![string is double -strict $actual_pl0_mhz] || abs($actual_pl0_mhz - 187.5) > 0.0001} {
        error "PS PL0 actual configured frequency differs from requested 187.5 MHz: $actual_pl0_mhz"
    }
    save_bd_design
    set bd [get_files pilot_bd.bd]
    generate_target all $bd
    add_files -norecurse [make_wrapper -files $bd -top]
    set_property top pilot_bd_wrapper [current_fileset]
    update_compile_order -fileset sources_1
}
timed synthesis {
    launch_runs synth_1 -jobs 2
    wait_on_run synth_1
    run_complete synth_1
}
timed route {
    launch_runs impl_1 -to_step route_design -jobs 2
    wait_on_run impl_1
    run_complete impl_1
}
timed route_acceptance {
    open_run impl_1
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output_root timing.rpt]
    report_timing -delay_type max -max_paths 10 -file [file join $output_root setup_paths.rpt]
    report_timing -delay_type min -max_paths 10 -file [file join $output_root hold_paths.rpt]
    report_utilization -file [file join $output_root utilization.rpt]
    report_utilization -hierarchical -file [file join $output_root utilization_hier.rpt]
    report_route_status -file [file join $output_root route_status.rpt]
    report_drc -ruledeck default -file [file join $output_root drc.rpt]
    foreach violation [get_drc_violations -quiet] {
        set severity [string tolower [get_property SEVERITY $violation]]
        if {$severity in {error fatal {critical warning}}} {error "DRC gate failed: $violation ($severity)"}
    }
    set route [read_text [file join $output_root route_status.rpt]]
    foreach {variable label} {routable {routable nets} fully_routed {fully routed nets} route_errors {nets with routing errors}} {
        if {![regexp "# of $label\\.+\\s*:\\s*(\[0-9\]+)" $route ignored $variable]} {error "missing route count: $label"}
    }
    if {$routable <= 0 || $routable != $fully_routed || $route_errors != 0} {error "route incomplete or erroneous"}
    set timing [read_text [file join $output_root timing.rpt]]
    foreach category {no_clock unconstrained_internal_endpoints} {
        if {![regexp "checking $category \\((\[0-9\]+)\\)" $timing ignored count] || $count != 0} {
            error "timing coverage gate failed: $category"
        }
    }
    set table_index [string first "WNS(ns)" $timing]
    if {$table_index < 0} {error "missing design timing summary"}
    set metrics {}
    foreach line [split [string range $timing $table_index end] \n] {
        set fields [regexp -all -inline {\S+} $line]
        if {[llength $fields] == 12 && [string is double -strict [lindex $fields 0]]} {set metrics $fields; break}
    }
    if {[llength $metrics] != 12} {error "unrecognized design timing summary"}
    lassign $metrics wns tns setup_failing setup_total whs ths hold_failing hold_total wpws tpws pulse_failing pulse_total
    if {$wns < 0 || $whs < 0 || $wpws < 0 || $tns != 0 || $ths != 0 || $tpws != 0 ||
        $setup_failing != 0 || $hold_failing != 0 || $pulse_failing != 0 || $setup_total <= 0 ||
        [string first "All user specified timing constraints are met." $timing] < 0} {error "timing gate failed"}
    set found_target_clock 0
    foreach clock [get_clocks -quiet] {
        if {abs([get_property PERIOD $clock] - 16.0/3.0) < 0.001} {set found_target_clock 1}
    }
    if {!$found_target_clock} {error "target 187.5 MHz clock absent from implemented timing constraints"}
}
timed bitstream {
    launch_runs impl_1 -to_step write_bitstream -jobs 2
    wait_on_run impl_1
    run_complete impl_1
}
timed artifact_export {
    set bit [one [glob -nocomplain [file join $output_root project pilot_shell.runs impl_1 pilot_bd_wrapper.bit]] "bitstream"]
    set hwhs {}
    foreach area {pilot_shell.gen pilot_shell.srcs} {
        foreach candidate [glob -nocomplain [file join $output_root project $area sources_1 bd pilot_bd hw_handoff pilot_bd.hwh]] {lappend hwhs $candidate}
    }
    set hwh [one $hwhs "hardware handoff"]
    file copy $bit [file join $output_root top.bit]
    file copy $hwh [file join $output_root top.hwh]
    set fd [open [file join $output_root shell_metadata.json] w]
    puts $fd [format {{"schema_version":3,"scope":"TEMPLATE_COVERAGE_ONLY","hardware_board_test":"NOT_RUN","part":"xczu7ev-ffvc1156-2-e","tool_version":%s,"board_preset":%s,"phase_pin":%s,"phase_valid_pin":%s,"phase_interface":"ap_vld_latched","clock_hz":187500000,"clock_period_ns":5.333333333333333,"hls_target_period_ns":5.0,"WNS_ns":%s,"WHS_ns":%s,"WPWS_ns":%s,"routable_nets":%s,"fully_routed_nets":%s,"routing_errors":%s,"route_timing_drc":"PASS","core_control_base":2684354560,"timer_counts_base":2684420096,"timer_sequence_base":2684485632,"timer_identity_base":2684551168,"gpio_data_offset":0,"gpio_data2_offset":8,"timer_magic":1345541169,"phase_counter_abi":"P3T1","phase_contract":"1_load_2_onchip_compute_3_store_repeated_tiles_0_idle","status_bits":{"valid":0,"busy":1,"counter_overflow":2,"phase_error":3,"sequence_overflow":4},"phase_semantics":{"load":1,"compute":2,"store":3,"idle":0},"reports":["timing.rpt","setup_paths.rpt","hold_paths.rpt","utilization.rpt","utilization_hier.rpt","route_status.rpt","drc.rpt"]}} \
        [json_quote [version -short]] [json_quote $board_part] [json_quote $phase_pin] [json_quote $phase_valid_pin] $wns $whs $wpws $routable $fully_routed $route_errors]
    close $fd
    set fd [open [file join $output_root ROUTE_TIMING_PASS] w]
    puts $fd "Template coverage integrated shell only; complete route, setup/hold/pulse timing, DRC gates passed; board evidence NOT_RUN."
    close $fd
}
close_project
exit
