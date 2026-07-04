module axis_byte_buffer_source #(
    parameter integer DATA_WIDTH = 8,
    parameter integer KEEP_WIDTH = (DATA_WIDTH + 7) / 8,
    parameter integer WORD_COUNT = 50176,
    parameter integer BYTE_COUNT = WORD_COUNT * KEEP_WIDTH,
    parameter integer ADDR_WIDTH = 16
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  byte_we,
    input  wire [ADDR_WIDTH-1:0] byte_addr,
    input  wire [7:0]            byte_data,
    input  wire                  start,
    output reg [DATA_WIDTH-1:0]  tdata,
    output reg [KEEP_WIDTH-1:0]  tkeep,
    output reg [KEEP_WIDTH-1:0]  tstrb,
    output reg                   tlast,
    output reg                   tvalid,
    input  wire                  tready,
    output reg                   done
);

reg [7:0] memory [0:BYTE_COUNT-1];
reg [31:0] word_index;
integer lane;

always @(*) begin
    tdata = {DATA_WIDTH{1'b0}};
    tkeep = {KEEP_WIDTH{1'b0}};
    tstrb = {KEEP_WIDTH{1'b0}};
    for (lane = 0; lane < KEEP_WIDTH; lane = lane + 1) begin
        if ((word_index * KEEP_WIDTH + lane) < BYTE_COUNT) begin
            tdata[lane*8 +: 8] = memory[word_index * KEEP_WIDTH + lane];
            tkeep[lane] = 1'b1;
            tstrb[lane] = 1'b1;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        word_index <= 0;
        tlast <= 0;
        tvalid <= 0;
        done <= 0;
    end else begin
        if (byte_we && byte_addr < BYTE_COUNT)
            memory[byte_addr] <= byte_data;
        if (start) begin
            word_index <= 0;
            tlast <= (WORD_COUNT == 1);
            tvalid <= 1'b1;
            done <= 1'b0;
        end else if (tvalid && tready) begin
            if (word_index + 1 >= WORD_COUNT) begin
                tvalid <= 1'b0;
                tlast <= 1'b0;
                done <= 1'b1;
            end else begin
                word_index <= word_index + 1'b1;
                tlast <= (word_index + 2 >= WORD_COUNT);
            end
        end
    end
end

endmodule
