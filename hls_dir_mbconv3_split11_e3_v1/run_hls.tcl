set script_dir [file dirname [file normalize [info script]]]
set project_dir $::env(DIR_V1_PROJECT_DIR)
set vector_dir $::env(DIR_V1_VECTOR_DIR)
set part $::env(DIR_V1_PART)
set clock_ns $::env(DIR_V1_CLOCK_NS)

set ::env(DIR_V1_VECTOR_DIR) [file normalize $vector_dir]
open_project -reset $project_dir
set_top dir_mbconv3_split11_e3_v1_int8
add_files [file join $script_dir dir_mbconv3_split11_e3_v1.cpp] -cflags "-std=c++11"
add_files -tb [file join $script_dir tb.cpp] -cflags "-std=c++11"
open_solution -reset solution1
set_part $part
create_clock -period $clock_ns -name default
csim_design
csynth_design
exit
