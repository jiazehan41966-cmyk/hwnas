set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 sim wrapper]
set project_dir [file join $result_root project]

file mkdir $result_root
set sim_part xczu7ev-ffvc1156-2-e
if {[llength [get_parts -quiet $sim_part]] == 0} {
    # Behavioral simulation is device independent. Falling back here does not
    # authorize synthesis or implementation for a non-ZCU104 part.
    set sim_part [lindex [get_parts] 0]
    puts "WARNING: ZCU104 device files unavailable; wrapper behavioral simulation uses $sim_part only."
}
create_project -force wrapper_sim $project_dir -part $sim_part
add_files [file join $script_dir operator_benchmark_wrapper.v]
add_files -fileset sim_1 [file join $script_dir tb_operator_benchmark_wrapper.sv]
set_property top tb_operator_benchmark_wrapper [get_filesets sim_1]
set_property xsim.simulate.runtime all [get_filesets sim_1]
launch_simulation -simset sim_1 -mode behavioral
close_sim
close_project
exit
