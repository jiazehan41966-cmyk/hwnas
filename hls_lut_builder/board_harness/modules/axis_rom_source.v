module axis_rom_source #(
    parameter integer DATA_WIDTH = 8,
    parameter integer KEEP_WIDTH = (DATA_WIDTH / 8),
    parameter integer WORD_COUNT = 1,
    parameter MEM_FILE_00 = "",
    parameter MEM_FILE_01 = "",
    parameter MEM_FILE_02 = "",
    parameter MEM_FILE_03 = "",
    parameter MEM_FILE_04 = "",
    parameter MEM_FILE_05 = "",
    parameter MEM_FILE_06 = "",
    parameter MEM_FILE_07 = "",
    parameter MEM_FILE_08 = "",
    parameter MEM_FILE_09 = "",
    parameter MEM_FILE_10 = "",
    parameter MEM_FILE_11 = "",
    parameter MEM_FILE_12 = "",
    parameter MEM_FILE_13 = "",
    parameter MEM_FILE_14 = "",
    parameter MEM_FILE_15 = "",
    parameter MEM_FILE_16 = "",
    parameter MEM_FILE_17 = "",
    parameter MEM_FILE_18 = "",
    parameter MEM_FILE_19 = "",
    parameter MEM_FILE_20 = "",
    parameter MEM_FILE_21 = "",
    parameter MEM_FILE_22 = "",
    parameter MEM_FILE_23 = "",
    parameter MEM_FILE_24 = "",
    parameter MEM_FILE_25 = "",
    parameter MEM_FILE_26 = "",
    parameter MEM_FILE_27 = "",
    parameter MEM_FILE_28 = "",
    parameter MEM_FILE_29 = "",
    parameter MEM_FILE_30 = "",
    parameter MEM_FILE_31 = ""
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    output reg [DATA_WIDTH-1:0]   tdata,
    output reg [KEEP_WIDTH-1:0]   tkeep,
    output reg [KEEP_WIDTH-1:0]   tstrb,
    output reg                    tlast,
    output reg                    tvalid,
    input  wire                   tready,
    output reg                    done
);

localparam integer ADDR_WIDTH = (WORD_COUNT <= 2) ? 1 : $clog2(WORD_COUNT);
localparam integer LANE_COUNT = KEEP_WIDTH;

reg [7:0] lane_rom_00 [0:WORD_COUNT-1];
reg [7:0] lane_rom_01 [0:WORD_COUNT-1];
reg [7:0] lane_rom_02 [0:WORD_COUNT-1];
reg [7:0] lane_rom_03 [0:WORD_COUNT-1];
reg [7:0] lane_rom_04 [0:WORD_COUNT-1];
reg [7:0] lane_rom_05 [0:WORD_COUNT-1];
reg [7:0] lane_rom_06 [0:WORD_COUNT-1];
reg [7:0] lane_rom_07 [0:WORD_COUNT-1];
reg [7:0] lane_rom_08 [0:WORD_COUNT-1];
reg [7:0] lane_rom_09 [0:WORD_COUNT-1];
reg [7:0] lane_rom_10 [0:WORD_COUNT-1];
reg [7:0] lane_rom_11 [0:WORD_COUNT-1];
reg [7:0] lane_rom_12 [0:WORD_COUNT-1];
reg [7:0] lane_rom_13 [0:WORD_COUNT-1];
reg [7:0] lane_rom_14 [0:WORD_COUNT-1];
reg [7:0] lane_rom_15 [0:WORD_COUNT-1];
reg [7:0] lane_rom_16 [0:WORD_COUNT-1];
reg [7:0] lane_rom_17 [0:WORD_COUNT-1];
reg [7:0] lane_rom_18 [0:WORD_COUNT-1];
reg [7:0] lane_rom_19 [0:WORD_COUNT-1];
reg [7:0] lane_rom_20 [0:WORD_COUNT-1];
reg [7:0] lane_rom_21 [0:WORD_COUNT-1];
reg [7:0] lane_rom_22 [0:WORD_COUNT-1];
reg [7:0] lane_rom_23 [0:WORD_COUNT-1];
reg [7:0] lane_rom_24 [0:WORD_COUNT-1];
reg [7:0] lane_rom_25 [0:WORD_COUNT-1];
reg [7:0] lane_rom_26 [0:WORD_COUNT-1];
reg [7:0] lane_rom_27 [0:WORD_COUNT-1];
reg [7:0] lane_rom_28 [0:WORD_COUNT-1];
reg [7:0] lane_rom_29 [0:WORD_COUNT-1];
reg [7:0] lane_rom_30 [0:WORD_COUNT-1];
reg [7:0] lane_rom_31 [0:WORD_COUNT-1];

reg [ADDR_WIDTH-1:0] index_reg = {ADDR_WIDTH{1'b0}};
reg active = 1'b0;
reg [DATA_WIDTH-1:0] current_word;

integer i;
initial begin
    for (i = 0; i < WORD_COUNT; i = i + 1) begin
        lane_rom_00[i] = 8'h00;
        lane_rom_01[i] = 8'h00;
        lane_rom_02[i] = 8'h00;
        lane_rom_03[i] = 8'h00;
        lane_rom_04[i] = 8'h00;
        lane_rom_05[i] = 8'h00;
        lane_rom_06[i] = 8'h00;
        lane_rom_07[i] = 8'h00;
        lane_rom_08[i] = 8'h00;
        lane_rom_09[i] = 8'h00;
        lane_rom_10[i] = 8'h00;
        lane_rom_11[i] = 8'h00;
        lane_rom_12[i] = 8'h00;
        lane_rom_13[i] = 8'h00;
        lane_rom_14[i] = 8'h00;
        lane_rom_15[i] = 8'h00;
        lane_rom_16[i] = 8'h00;
        lane_rom_17[i] = 8'h00;
        lane_rom_18[i] = 8'h00;
        lane_rom_19[i] = 8'h00;
        lane_rom_20[i] = 8'h00;
        lane_rom_21[i] = 8'h00;
        lane_rom_22[i] = 8'h00;
        lane_rom_23[i] = 8'h00;
        lane_rom_24[i] = 8'h00;
        lane_rom_25[i] = 8'h00;
        lane_rom_26[i] = 8'h00;
        lane_rom_27[i] = 8'h00;
        lane_rom_28[i] = 8'h00;
        lane_rom_29[i] = 8'h00;
        lane_rom_30[i] = 8'h00;
        lane_rom_31[i] = 8'h00;
    end
    if (MEM_FILE_00 != "") $readmemh(MEM_FILE_00, lane_rom_00);
    if (MEM_FILE_01 != "") $readmemh(MEM_FILE_01, lane_rom_01);
    if (MEM_FILE_02 != "") $readmemh(MEM_FILE_02, lane_rom_02);
    if (MEM_FILE_03 != "") $readmemh(MEM_FILE_03, lane_rom_03);
    if (MEM_FILE_04 != "") $readmemh(MEM_FILE_04, lane_rom_04);
    if (MEM_FILE_05 != "") $readmemh(MEM_FILE_05, lane_rom_05);
    if (MEM_FILE_06 != "") $readmemh(MEM_FILE_06, lane_rom_06);
    if (MEM_FILE_07 != "") $readmemh(MEM_FILE_07, lane_rom_07);
    if (MEM_FILE_08 != "") $readmemh(MEM_FILE_08, lane_rom_08);
    if (MEM_FILE_09 != "") $readmemh(MEM_FILE_09, lane_rom_09);
    if (MEM_FILE_10 != "") $readmemh(MEM_FILE_10, lane_rom_10);
    if (MEM_FILE_11 != "") $readmemh(MEM_FILE_11, lane_rom_11);
    if (MEM_FILE_12 != "") $readmemh(MEM_FILE_12, lane_rom_12);
    if (MEM_FILE_13 != "") $readmemh(MEM_FILE_13, lane_rom_13);
    if (MEM_FILE_14 != "") $readmemh(MEM_FILE_14, lane_rom_14);
    if (MEM_FILE_15 != "") $readmemh(MEM_FILE_15, lane_rom_15);
    if (MEM_FILE_16 != "") $readmemh(MEM_FILE_16, lane_rom_16);
    if (MEM_FILE_17 != "") $readmemh(MEM_FILE_17, lane_rom_17);
    if (MEM_FILE_18 != "") $readmemh(MEM_FILE_18, lane_rom_18);
    if (MEM_FILE_19 != "") $readmemh(MEM_FILE_19, lane_rom_19);
    if (MEM_FILE_20 != "") $readmemh(MEM_FILE_20, lane_rom_20);
    if (MEM_FILE_21 != "") $readmemh(MEM_FILE_21, lane_rom_21);
    if (MEM_FILE_22 != "") $readmemh(MEM_FILE_22, lane_rom_22);
    if (MEM_FILE_23 != "") $readmemh(MEM_FILE_23, lane_rom_23);
    if (MEM_FILE_24 != "") $readmemh(MEM_FILE_24, lane_rom_24);
    if (MEM_FILE_25 != "") $readmemh(MEM_FILE_25, lane_rom_25);
    if (MEM_FILE_26 != "") $readmemh(MEM_FILE_26, lane_rom_26);
    if (MEM_FILE_27 != "") $readmemh(MEM_FILE_27, lane_rom_27);
    if (MEM_FILE_28 != "") $readmemh(MEM_FILE_28, lane_rom_28);
    if (MEM_FILE_29 != "") $readmemh(MEM_FILE_29, lane_rom_29);
    if (MEM_FILE_30 != "") $readmemh(MEM_FILE_30, lane_rom_30);
    if (MEM_FILE_31 != "") $readmemh(MEM_FILE_31, lane_rom_31);
end

function [DATA_WIDTH-1:0] read_word;
    input [ADDR_WIDTH-1:0] idx;
    reg [DATA_WIDTH-1:0] assembled;
begin
    assembled = {DATA_WIDTH{1'b0}};
    if (LANE_COUNT > 0) assembled[7:0]     = lane_rom_00[idx];
    if (LANE_COUNT > 1) assembled[15:8]    = lane_rom_01[idx];
    if (LANE_COUNT > 2) assembled[23:16]   = lane_rom_02[idx];
    if (LANE_COUNT > 3) assembled[31:24]   = lane_rom_03[idx];
    if (LANE_COUNT > 4) assembled[39:32]   = lane_rom_04[idx];
    if (LANE_COUNT > 5) assembled[47:40]   = lane_rom_05[idx];
    if (LANE_COUNT > 6) assembled[55:48]   = lane_rom_06[idx];
    if (LANE_COUNT > 7) assembled[63:56]   = lane_rom_07[idx];
    if (LANE_COUNT > 8) assembled[71:64]   = lane_rom_08[idx];
    if (LANE_COUNT > 9) assembled[79:72]   = lane_rom_09[idx];
    if (LANE_COUNT > 10) assembled[87:80]  = lane_rom_10[idx];
    if (LANE_COUNT > 11) assembled[95:88]  = lane_rom_11[idx];
    if (LANE_COUNT > 12) assembled[103:96] = lane_rom_12[idx];
    if (LANE_COUNT > 13) assembled[111:104] = lane_rom_13[idx];
    if (LANE_COUNT > 14) assembled[119:112] = lane_rom_14[idx];
    if (LANE_COUNT > 15) assembled[127:120] = lane_rom_15[idx];
    if (LANE_COUNT > 16) assembled[135:128] = lane_rom_16[idx];
    if (LANE_COUNT > 17) assembled[143:136] = lane_rom_17[idx];
    if (LANE_COUNT > 18) assembled[151:144] = lane_rom_18[idx];
    if (LANE_COUNT > 19) assembled[159:152] = lane_rom_19[idx];
    if (LANE_COUNT > 20) assembled[167:160] = lane_rom_20[idx];
    if (LANE_COUNT > 21) assembled[175:168] = lane_rom_21[idx];
    if (LANE_COUNT > 22) assembled[183:176] = lane_rom_22[idx];
    if (LANE_COUNT > 23) assembled[191:184] = lane_rom_23[idx];
    if (LANE_COUNT > 24) assembled[199:192] = lane_rom_24[idx];
    if (LANE_COUNT > 25) assembled[207:200] = lane_rom_25[idx];
    if (LANE_COUNT > 26) assembled[215:208] = lane_rom_26[idx];
    if (LANE_COUNT > 27) assembled[223:216] = lane_rom_27[idx];
    if (LANE_COUNT > 28) assembled[231:224] = lane_rom_28[idx];
    if (LANE_COUNT > 29) assembled[239:232] = lane_rom_29[idx];
    if (LANE_COUNT > 30) assembled[247:240] = lane_rom_30[idx];
    if (LANE_COUNT > 31) assembled[255:248] = lane_rom_31[idx];
    read_word = assembled;
end
endfunction

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        index_reg <= {ADDR_WIDTH{1'b0}};
        active <= 1'b0;
        tvalid <= 1'b0;
        tdata <= {DATA_WIDTH{1'b0}};
        current_word <= {DATA_WIDTH{1'b0}};
        tkeep <= {KEEP_WIDTH{1'b1}};
        tstrb <= {KEEP_WIDTH{1'b1}};
        tlast <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        tkeep <= {KEEP_WIDTH{1'b1}};
        tstrb <= {KEEP_WIDTH{1'b1}};

        if (start) begin
            index_reg <= {ADDR_WIDTH{1'b0}};
            active <= 1'b1;
            tvalid <= 1'b1;
            current_word <= read_word({ADDR_WIDTH{1'b0}});
            tdata <= read_word({ADDR_WIDTH{1'b0}});
            tlast <= (WORD_COUNT == 1);
        end else if (active && tvalid && tready) begin
            if (index_reg == WORD_COUNT - 1) begin
                active <= 1'b0;
                tvalid <= 1'b0;
                tlast <= 1'b0;
                done <= 1'b1;
            end else begin
                index_reg <= index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                current_word <= read_word(index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1});
                tdata <= read_word(index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1});
                tlast <= ((index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1}) == WORD_COUNT - 1);
            end
        end
    end
end

endmodule
