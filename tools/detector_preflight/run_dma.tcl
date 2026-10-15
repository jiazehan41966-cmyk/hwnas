open_project hls_project
set_top h2_dma
add_files h2_dma.cpp -cflags {-std=c++14 -I.}
add_files -tb tb_dma.cpp -cflags {-std=c++14 -I.}
foreach dir {pw fpn} {add_files -tb $dir}
open_solution solution1 -flow_target vivado
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 5
config_interface -m_axi_addr64
csim_design -O
csynth_design
export_design -format ip_catalog -vendor hwnas.org -library hls -version 1.0
exit
