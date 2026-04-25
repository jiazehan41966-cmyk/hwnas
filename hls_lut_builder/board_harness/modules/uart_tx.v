module uart_tx #(
    parameter integer CLK_FREQ_HZ = 200000000,
    parameter integer BAUD = 115200
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       valid,
    output wire       ready,
    output reg        tx,
    output reg        busy
);

localparam integer BAUD_DIV = (CLK_FREQ_HZ / BAUD);

reg [31:0] baud_counter = 32'd0;
reg [3:0] bit_index = 4'd0;
reg [9:0] shift_reg = 10'h3ff;

assign ready = !busy;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        baud_counter <= 32'd0;
        bit_index <= 4'd0;
        shift_reg <= 10'h3ff;
        tx <= 1'b1;
        busy <= 1'b0;
    end else begin
        if (!busy) begin
            tx <= 1'b1;
            if (valid) begin
                shift_reg <= {1'b1, data, 1'b0};
                bit_index <= 4'd0;
                baud_counter <= 32'd0;
                busy <= 1'b1;
                tx <= 1'b0;
            end
        end else begin
            if (baud_counter == BAUD_DIV - 1) begin
                baud_counter <= 32'd0;
                bit_index <= bit_index + 4'd1;
                shift_reg <= {1'b1, shift_reg[9:1]};
                tx <= shift_reg[1];
                if (bit_index == 4'd9) begin
                    busy <= 1'b0;
                    tx <= 1'b1;
                end
            end else begin
                baud_counter <= baud_counter + 32'd1;
            end
        end
    end
end

endmodule
