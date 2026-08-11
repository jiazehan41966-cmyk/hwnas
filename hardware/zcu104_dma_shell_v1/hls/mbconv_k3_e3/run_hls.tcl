set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_mbconv_k3_e3_pilot_v1 hls]
set project_dir [file join $result_root project]
set source_file [file join $script_dir src zcu104_stage4_k3_e3_kernel.cpp]
set include_dir [file join $script_dir include]
set part_name {xczu7ev-ffvc1156-2-e}

file mkdir $result_root
cd $result_root
# Vitis HLS 2022.2 rejects an absolute path as the project name.  Keep the
# artifact path fixed by changing into result_root, then open a relative name.
open_project -reset project
set_top zcu104_stage4_k3_e3_kernel
add_files $source_file -cflags "-std=c++14 -O3 -I$include_dir"
open_solution -reset solution1 -flow_target vivado
set_part $part_name
create_clock -period 5.000 -name default
set_clock_uncertainty 0.200

csynth_design
export_design -format ip_catalog -rtl verilog -vendor hwnas.org -library hls -version 1.0 \
    -description {Frozen Stage4 MBConv-K3-E3 core for the ZCU104 DMA shell pilot}

set report_file [file join $project_dir solution1 syn report zcu104_stage4_k3_e3_kernel_csynth.rpt]
set component_xml [file join $project_dir solution1 impl ip component.xml]
if {![file exists $report_file]} {
    error "Missing HLS synthesis report: $report_file"
}
if {![file exists $component_xml]} {
    error "Missing exported HLS component.xml: $component_xml"
}

set xml_handle [open $component_xml r]
set xml_text [read $xml_handle]
close $xml_handle
foreach expected_text {hwnas.org hls zcu104_stage4_k3_e3_kernel 1.0 input_stream output_stream weights biases ap_ctrl} {
    if {[string first $expected_text $xml_text] < 0} {
        error "Exported component.xml is missing expected text: $expected_text"
    }
}

set result_file [open [file join $result_root export_result.json] w]
puts $result_file "{"
puts $result_file "  \"schema_version\": 1,"
puts $result_file "  \"status\": \"PASS_HLS_SYNTHESIS_AND_IP_EXPORT\","
puts $result_file "  \"toolchain\": \"Vitis HLS 2022.2\","
puts $result_file "  \"target_part\": \"$part_name\","
puts $result_file "  \"clock_period_ns\": 5.0,"
puts $result_file "  \"operator\": \"stage4_mbconv_k3_e3\","
puts $result_file "  \"vlnv\": \"hwnas.org:hls:zcu104_stage4_k3_e3_kernel:1.0\","
puts $result_file "  \"component_xml\": \"$component_xml\","
puts $result_file "  \"mathematical_semantics\": \"frozen calculation; control changed to direct ap_ctrl_hs and parameter BRAMs constrained to ram_1p\","
puts $result_file "  \"claim_boundary\": \"HLS synthesis and IP export only; not Vivado implementation or board execution.\""
puts $result_file "}"
close $result_file

exit
