module result_uart_tx #(
    parameter integer CLK_FREQ_HZ = 200000000,
    parameter integer BAUD = 115200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,
    input  wire [7:0] status_code,
    input  wire [31:0] cycles,
    input  wire [31:0] checksum,
    input  wire [31:0] word_count,
    output wire       tx,
    output reg        busy,
    output reg        done
);

localparam integer FRAME_BYTES = 15;

reg [7:0] frame_mem [0:FRAME_BYTES-1];
reg [4:0] byte_index = 5'd0;
reg tx_valid = 1'b0;
reg [7:0] tx_data = 8'd0;
wire tx_ready;

uart_tx #(
    .CLK_FREQ_HZ(CLK_FREQ_HZ),
    .BAUD(BAUD)
) u_uart_tx (
    .clk(clk),
    .rst_n(rst_n),
    .data(tx_data),
    .valid(tx_valid),
    .ready(tx_ready),
    .tx(tx),
    .busy()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 1'b0;
        done <= 1'b0;
        byte_index <= 5'd0;
        tx_valid <= 1'b0;
        tx_data <= 8'd0;
    end else begin
        done <= 1'b0;

        if (start && !busy) begin
            frame_mem[0]  <= 8'hA5;
            frame_mem[1]  <= 8'h5A;
            frame_mem[2]  <= status_code;
            frame_mem[3]  <= cycles[7:0];
            frame_mem[4]  <= cycles[15:8];
            frame_mem[5]  <= cycles[23:16];
            frame_mem[6]  <= cycles[31:24];
            frame_mem[7]  <= checksum[7:0];
            frame_mem[8]  <= checksum[15:8];
            frame_mem[9]  <= checksum[23:16];
            frame_mem[10] <= checksum[31:24];
            frame_mem[11] <= word_count[7:0];
            frame_mem[12] <= word_count[15:8];
            frame_mem[13] <= word_count[23:16];
            frame_mem[14] <= word_count[31:24];
            byte_index <= 5'd0;
            busy <= 1'b1;
            tx_data <= 8'hA5;
            tx_valid <= 1'b1;
        end else if (busy) begin
            if (tx_valid && tx_ready) begin
                if (byte_index == FRAME_BYTES - 1) begin
                    tx_valid <= 1'b0;
                    busy <= 1'b0;
                    done <= 1'b1;
                    byte_index <= 5'd0;
                end else begin
                    byte_index <= byte_index + 5'd1;
                    tx_data <= frame_mem[byte_index + 5'd1];
                    tx_valid <= 1'b1;
                end
            end else if (!tx_valid && tx_ready) begin
                tx_data <= frame_mem[byte_index];
                tx_valid <= 1'b1;
            end
        end else begin
            tx_valid <= 1'b0;
        end
    end
end

endmodule
