# Isolated synthesis audit for the 224x224 byte-wide dynamic input buffer.
set script_dir [file dirname [file normalize [info script]]]
set module_file [file normalize [file join $script_dir .. modules axis_byte_buffer_source.v]]
set output_dir [file normalize [lindex $argv 0]]
if {$output_dir eq ""} {
    set output_dir [file normalize [file join $script_dir .. results dynamic_buffer_synth_check]]
}
set word_count [lindex $argv 1]
if {$word_count eq ""} {
    # Fast inference smoke. Full 50176-byte synthesis belongs to the complete
    # harness build, where timing/resource evidence is recorded together.
    set word_count 4096
}
file mkdir $output_dir

read_verilog $module_file
synth_design -top axis_byte_buffer_source -part xc7k325t-ffg900-2 \
    -generic WORD_COUNT=$word_count
report_utilization -file [file join $output_dir utilization.rpt]
write_checkpoint -force [file join $output_dir axis_byte_buffer_source.dcp]
exit
