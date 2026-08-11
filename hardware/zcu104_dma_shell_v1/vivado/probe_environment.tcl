set parts [get_parts -quiet]
puts "INSTALLED_PART_COUNT=[llength $parts]"
puts "FIRST_INSTALLED_PART=[lindex $parts 0]"
puts "ZCU104_TARGET_MATCHES=[llength [get_parts -quiet xczu7ev-ffvc1156-2-e]]"
set board_repo {F:/vivado/Vivado/2023.2/data/xhub/boards/XilinxBoardStore/boards/Xilinx}
if {[file isdirectory $board_repo]} {
    set_param board.repoPaths [list $board_repo]
}
puts "ZCU104_BOARD_MATCHES=[llength [get_board_parts -quiet xilinx.com:zcu104:part0:1.1]]"
exit
