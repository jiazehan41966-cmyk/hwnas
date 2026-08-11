set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 sim wrapper_hls_core]
set project_dir [file join $result_root project]
set hls_rtl_dir [file join $repo_root results zcu104_dma_shell_v1 hls skip_core_portable_preflight project solution1 impl ip hdl verilog]

if {![file isdirectory $hls_rtl_dir]} {
    error "Missing exported portable HLS RTL at $hls_rtl_dir."
}
set installed_parts [get_parts -quiet]
if {[llength $installed_parts] == 0} {
    error "No Vivado part is installed for behavioral simulation project creation."
}
set simulation_part [lindex $installed_parts 0]

file mkdir $result_root
create_project -force wrapper_hls_core_sim $project_dir -part $simulation_part
add_files [file join $script_dir operator_benchmark_wrapper.v]
add_files [glob -nocomplain [file join $hls_rtl_dir *.v]]
add_files -fileset sim_1 [file join $script_dir tb_operator_benchmark_wrapper.sv]
set_property top tb_operator_benchmark_wrapper [get_filesets sim_1]
set_property verilog_define {ACTUAL_HLS_CORE} [get_filesets sim_1]
set_property xsim.simulate.runtime all [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
close_sim
close_project
exit
