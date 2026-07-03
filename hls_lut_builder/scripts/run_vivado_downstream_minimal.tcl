set argc_expected 5
if {$argc < $argc_expected} {
    puts "Usage: vivado -mode batch -source run_vivado_downstream_minimal.tcl -tclargs <rtl_dir> <top> <part> <clock_ns> <out_dir>"
    exit 1
}

set rtl_dir [file normalize [lindex $argv 0]]
set top_name [lindex $argv 1]
set part_name [lindex $argv 2]
set clock_ns [lindex $argv 3]
set out_dir [file normalize [lindex $argv 4]]
set report_dir [file join $out_dir reports]
set checkpoint_dir [file join $out_dir checkpoints]

file mkdir $out_dir
file mkdir $report_dir
file mkdir $checkpoint_dir

puts "=== Minimal Vivado downstream flow ==="
puts "RTL dir: $rtl_dir"
puts "Top: $top_name"
puts "Part: $part_name"
puts "Clock: $clock_ns ns"
puts "Output dir: $out_dir"

set rtl_files [lsort [glob -nocomplain -directory $rtl_dir *.v]]
if {[llength $rtl_files] == 0} {
    puts "ERROR: No Verilog files found under $rtl_dir"
    exit 2
}

foreach rtl_file $rtl_files {
    read_verilog $rtl_file
}

proc disable_parallel_helper_spawn {} {
    if {[llength [info commands rt::set_parameter]] > 0} {
        if {[catch {rt::set_parameter enableParallelHelperSpawn 0} err]} {
            puts "WARNING: could not disable Vivado parallel helper spawn: $err"
        } else {
            puts "Disabled Vivado parallel helper spawn via rt::set_parameter"
        }
    } else {
        puts "Vivado rt::set_parameter is not available before synth_design"
    }
}
disable_parallel_helper_spawn

if {[catch {synth_design -top $top_name -part $part_name -mode out_of_context} err]} {
    puts "ERROR: synth_design failed: $err"
    exit 3
}

create_clock -period $clock_ns -name ap_clk [get_ports ap_clk]
set_false_path -from [get_ports ap_rst_n]

write_checkpoint -force [file join $checkpoint_dir post_synth.dcp]
report_utilization -file [file join $report_dir post_synth_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $report_dir post_synth_timing_summary.rpt]

if {[catch {opt_design} err]} {
    puts "ERROR: opt_design failed: $err"
    exit 4
}
if {[catch {place_design} err]} {
    puts "ERROR: place_design failed: $err"
    exit 5
}
if {[catch {route_design} err]} {
    puts "ERROR: route_design failed: $err"
    exit 6
}

write_checkpoint -force [file join $checkpoint_dir post_route.dcp]
report_utilization -file [file join $report_dir post_route_utilization.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -file [file join $report_dir post_route_timing_summary.rpt]
if {[catch {report_power -file [file join $report_dir post_route_power.rpt]} err]} {
    puts "WARNING: report_power failed: $err"
}

puts "=== Minimal Vivado downstream flow completed successfully ==="
exit 0
