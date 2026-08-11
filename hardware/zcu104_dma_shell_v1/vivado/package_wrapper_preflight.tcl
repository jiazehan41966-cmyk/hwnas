set script_dir [file dirname [file normalize [info script]]]
set repo_root [file normalize [file join $script_dir .. .. ..]]
set result_root [file join $repo_root results zcu104_dma_shell_v1 vivado wrapper_package_preflight]
set wrapper_ip_root [file join $result_root ip]
set wrapper_rtl [file join $repo_root hardware zcu104_dma_shell_v1 rtl operator_benchmark_wrapper.v]
set preflight_part [lindex [get_parts] 0]

file mkdir $result_root
create_project -force wrapper_ip_pack [file join $result_root project] -part $preflight_part
add_files $wrapper_rtl
set_property top operator_benchmark_wrapper [current_fileset]
update_compile_order -fileset sources_1
ipx::package_project -root_dir $wrapper_ip_root -vendor hwnas.org -library user -taxonomy /HWNAS -import_files -set_current true
set wrapper_core [ipx::current_core]
set_property name operator_benchmark_wrapper $wrapper_core
set_property display_name {HWNAS Operator Benchmark Wrapper} $wrapper_core
set_property description {AXI DMA framing, control, status, and operator-core latency counter} $wrapper_core
set_property version 1.0 $wrapper_core

set expected_busifs {S_AXI_CTRL S_AXIS_DMA M_AXIS_DMA M_AXIS_CORE S_AXIS_CORE}
foreach busif $expected_busifs {
    if {[llength [ipx::get_bus_interfaces $busif -of_objects $wrapper_core]] != 1} {
        error "Expected bus interface $busif was not inferred."
    }
}

set ctrl_if [ipx::get_bus_interfaces S_AXI_CTRL -of_objects $wrapper_core]
set memory_maps [ipx::get_memory_maps -of_objects $wrapper_core]
if {[llength $memory_maps] != 1} {
    error "Expected exactly one inferred AXI-Lite memory map, found [llength $memory_maps]."
}
set memory_map [lindex $memory_maps 0]
set address_blocks [ipx::get_address_blocks -of_objects $memory_map]
if {[llength $address_blocks] != 1} {
    error "Expected exactly one inferred AXI-Lite address block, found [llength $address_blocks]."
}
set address_block [lindex $address_blocks 0]
set_property range 4096 $address_block
set_property width 32 $address_block
set_property usage register $address_block
set_property slave_memory_map_ref S_AXI_CTRL $ctrl_if

foreach register_spec {
    {CONTROL 0x00 read-write}
    {STATUS 0x04 read-only}
    {EXECUTION_COUNT_LO 0x08 read-only}
    {EXECUTION_COUNT_HI 0x0c read-only}
    {OPERATOR_CORE_LATENCY_CYCLES_LO 0x10 read-only}
    {OPERATOR_CORE_LATENCY_CYCLES_HI 0x14 read-only}
    {EXPECTED_INPUT_WORDS 0x18 read-only}
    {EXPECTED_OUTPUT_WORDS 0x1c read-only}
    {INPUT_WORDS_SEEN 0x20 read-only}
    {OUTPUT_WORDS_SEEN 0x24 read-only}
    {ERROR_FLAGS 0x28 read-only}
    {SHELL_ID 0x2c read-only}
    {VERSION 0x30 read-only}
    {OPERATOR_ID 0x34 read-only}
    {AXIS_DATA_BITS 0x38 read-only}
    {CLOCK_HZ 0x3c read-only}
} {
    lassign $register_spec reg_name reg_offset reg_access
    set reg_obj [ipx::add_register $reg_name $address_block]
    set_property address_offset $reg_offset $reg_obj
    set_property size 32 $reg_obj
    set_property access $reg_access $reg_obj
}

foreach busif $expected_busifs {
    ipx::associate_bus_interfaces -busif $busif -clock aclk $wrapper_core
}
ipx::create_xgui_files $wrapper_core
ipx::update_checksums $wrapper_core
ipx::check_integrity -quiet $wrapper_core
ipx::save_core $wrapper_core

puts "PASS: wrapper IP packaged with five AXI interfaces and one 4 KiB register block"
close_project
exit
