module uart_rx_byte #(
    parameter integer CLK_FREQ_HZ = 200000000,
    parameter integer BAUD = 921600
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       uart_rx,
    output reg        byte_valid,
    output reg [7:0]  byte_data,
    output reg        framing_error
);

localparam integer CLKS_PER_BIT = CLK_FREQ_HZ / BAUD;
localparam integer HALF_BIT = CLKS_PER_BIT / 2;

reg [31:0] clock_count;
reg [3:0] bit_index;
reg [7:0] shift;
reg [1:0] state;

localparam [1:0] IDLE = 2'd0;
localparam [1:0] START = 2'd1;
localparam [1:0] DATA = 2'd2;
localparam [1:0] STOP = 2'd3;

always @(posedge clk) begin
    if (rst) begin
        clock_count <= 0;
        bit_index <= 0;
        shift <= 0;
        state <= IDLE;
        byte_valid <= 0;
        byte_data <= 0;
        framing_error <= 0;
    end else begin
        byte_valid <= 0;
        framing_error <= 0;
        case (state)
            IDLE: begin
                clock_count <= 0;
                bit_index <= 0;
                if (!uart_rx)
                    state <= START;
            end
            START: begin
                if (clock_count >= HALF_BIT) begin
                    clock_count <= 0;
                    if (!uart_rx)
                        state <= DATA;
                    else
                        state <= IDLE;
                end else begin
                    clock_count <= clock_count + 1'b1;
                end
            end
            DATA: begin
                if (clock_count >= CLKS_PER_BIT - 1) begin
                    clock_count <= 0;
                    shift[bit_index] <= uart_rx;
                    if (bit_index == 7) begin
                        bit_index <= 0;
                        state <= STOP;
                    end else begin
                        bit_index <= bit_index + 1'b1;
                    end
                end else begin
                    clock_count <= clock_count + 1'b1;
                end
            end
            STOP: begin
                if (clock_count >= CLKS_PER_BIT - 1) begin
                    clock_count <= 0;
                    state <= IDLE;
                    if (uart_rx) begin
                        byte_data <= shift;
                        byte_valid <= 1'b1;
                    end else begin
                        framing_error <= 1'b1;
                    end
                end else begin
                    clock_count <= clock_count + 1'b1;
                end
            end
            default: state <= IDLE;
        endcase
    end
end

endmodule
