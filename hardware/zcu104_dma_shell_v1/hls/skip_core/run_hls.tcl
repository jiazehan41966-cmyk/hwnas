set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 hls skip_core]
set project_dir [file join $result_root project]

file mkdir $result_root
open_project -reset $project_dir
set_top zcu104_stage4_skip_core
add_files [file join $script_dir zcu104_stage4_skip_core.cpp] -cflags "-std=c++14"
add_files -tb [file join $script_dir tb.cpp] -cflags "-std=c++14"

open_solution -reset solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5.000 -name default

csim_design
csynth_design
export_design -format ip_catalog -rtl verilog -vendor hwnas.org -library hls -version 1.0 \
    -description {Frozen Stage4 Skip core for the ZCU104 DMA micro-benchmark shell}

exit
