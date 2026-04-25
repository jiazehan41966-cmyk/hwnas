module axis_checksum_sink #(
    parameter integer DATA_WIDTH = 8,
    parameter integer KEEP_WIDTH = (DATA_WIDTH / 8),
    parameter integer EXPECTED_WORD_COUNT = 1
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   clear,
    input  wire [DATA_WIDTH-1:0]  tdata,
    input  wire [KEEP_WIDTH-1:0]  tkeep,
    input  wire [KEEP_WIDTH-1:0]  tstrb,
    input  wire                   tlast,
    input  wire                   tvalid,
    output wire                   tready,
    output reg                    done,
    output reg [31:0]             checksum,
    output reg [31:0]             word_count
);

integer lane;
reg [31:0] checksum_next;

assign tready = 1'b1;

always @(*) begin
    checksum_next = checksum;
    for (lane = 0; lane < DATA_WIDTH; lane = lane + 32) begin
        checksum_next = checksum_next ^ tdata[lane +: 32];
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        done <= 1'b0;
        checksum <= 32'd0;
        word_count <= 32'd0;
    end else begin
        if (clear) begin
            done <= 1'b0;
            checksum <= 32'd0;
            word_count <= 32'd0;
        end else if (tvalid && tready) begin
            checksum <= checksum_next;
            word_count <= word_count + 32'd1;
            if (tlast || (word_count + 32'd1 == EXPECTED_WORD_COUNT)) begin
                done <= 1'b1;
            end
        end
    end
end

endmodule
