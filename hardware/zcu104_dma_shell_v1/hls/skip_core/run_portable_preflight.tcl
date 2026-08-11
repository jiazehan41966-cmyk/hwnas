set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 hls skip_core_portable_preflight]
set project_dir [file join $result_root project]

file mkdir $result_root
open_project -reset $project_dir
set_top zcu104_stage4_skip_core
add_files [file join $script_dir zcu104_stage4_skip_core.cpp] -cflags "-std=c++14"
add_files -tb [file join $script_dir tb.cpp] -cflags "-std=c++14"

open_solution -reset solution1 -flow_target vivado
# Functional/synthesizability preflight only. This installed part must never be
# reported as ZCU104 evidence; run_hls.tcl remains the sole target export path.
set_part {xc7k70tfbv676-1}
create_clock -period 5.000 -name default
csim_design
csynth_design
export_design -format ip_catalog -rtl verilog -vendor hwnas.org -library preflight -version 1.0 \
    -description {NON-ZCU104 portable preflight for the Stage4 Skip core}
exit
