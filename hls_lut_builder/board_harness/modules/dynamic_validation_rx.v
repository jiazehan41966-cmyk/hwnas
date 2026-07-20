module dynamic_validation_rx #(
    parameter integer MAX_PAYLOAD_BYTES = 50176,
    parameter integer ADDR_WIDTH = 16
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  rx_byte_valid,
    input  wire [7:0]            rx_byte,
    output reg                   payload_we,
    output reg [ADDR_WIDTH-1:0]  payload_addr,
    output reg [7:0]             payload_data,
    output reg                   command_valid,
    output reg [7:0]             command,
    output reg [31:0]            sample_id,
    output reg [31:0]            payload_len,
    output reg [31:0]            repeat_count,
    output reg                   crc_ok,
    output reg [7:0]             error_code
);

localparam [7:0] VERSION = 8'd1;
localparam [3:0] S_MAGIC0 = 4'd0;
localparam [3:0] S_MAGIC1 = 4'd1;
localparam [3:0] S_VERSION = 4'd2;
localparam [3:0] S_COMMAND = 4'd3;
localparam [3:0] S_SAMPLE = 4'd4;
localparam [3:0] S_LENGTH = 4'd5;
localparam [3:0] S_PAYLOAD = 4'd6;
localparam [3:0] S_CRC = 4'd7;

reg [3:0] state;
reg [2:0] field_index;
reg [31:0] payload_index;
reg [31:0] crc_state;
reg [31:0] received_crc;

function [31:0] crc32_byte;
    input [31:0] crc_in;
    input [7:0] data;
    integer bit_index;
    reg [31:0] crc;
    begin
        crc = crc_in ^ data;
        for (bit_index = 0; bit_index < 8; bit_index = bit_index + 1) begin
            if (crc[0])
                crc = (crc >> 1) ^ 32'hEDB88320;
            else
                crc = crc >> 1;
        end
        crc32_byte = crc;
    end
endfunction

wire [31:0] sample_with_byte =
    sample_id | ({24'd0, rx_byte} << (field_index * 8));
wire [31:0] length_with_byte =
    payload_len | ({24'd0, rx_byte} << (field_index * 8));
wire [31:0] received_crc_with_byte =
    received_crc | ({24'd0, rx_byte} << (field_index * 8));

always @(posedge clk) begin
    if (rst) begin
        state <= S_MAGIC0;
        field_index <= 0;
        payload_index <= 0;
        crc_state <= 32'hFFFFFFFF;
        received_crc <= 0;
        payload_we <= 0;
        payload_addr <= 0;
        payload_data <= 0;
        command_valid <= 0;
        command <= 0;
        sample_id <= 0;
        payload_len <= 0;
        repeat_count <= 0;
        crc_ok <= 0;
        error_code <= 0;
    end else begin
        payload_we <= 0;
        command_valid <= 0;
        if (rx_byte_valid) begin
            case (state)
                S_MAGIC0: begin
                    if (rx_byte == 8'hA5)
                        state <= S_MAGIC1;
                end
                S_MAGIC1: begin
                    if (rx_byte == 8'h5A) begin
                        state <= S_VERSION;
                        crc_state <= 32'hFFFFFFFF;
                        sample_id <= 0;
                        payload_len <= 0;
                        repeat_count <= 0;
                        error_code <= 0;
                        crc_ok <= 0;
                    end else if (rx_byte != 8'hA5) begin
                        state <= S_MAGIC0;
                    end
                end
                S_VERSION: begin
                    crc_state <= crc32_byte(crc_state, rx_byte);
                    if (rx_byte != VERSION) begin
                        error_code <= 8'd1;
                        state <= S_MAGIC0;
                    end else begin
                        state <= S_COMMAND;
                    end
                end
                S_COMMAND: begin
                    command <= rx_byte;
                    crc_state <= crc32_byte(crc_state, rx_byte);
                    field_index <= 0;
                    state <= S_SAMPLE;
                end
                S_SAMPLE: begin
                    sample_id <= sample_with_byte;
                    crc_state <= crc32_byte(crc_state, rx_byte);
                    if (field_index == 3) begin
                        field_index <= 0;
                        state <= S_LENGTH;
                    end else begin
                        field_index <= field_index + 1'b1;
                    end
                end
                S_LENGTH: begin
                    payload_len <= length_with_byte;
                    crc_state <= crc32_byte(crc_state, rx_byte);
                    if (field_index == 3) begin
                        field_index <= 0;
                        payload_index <= 0;
                        received_crc <= 0;
                        if (length_with_byte > MAX_PAYLOAD_BYTES) begin
                            error_code <= 8'd2;
                            state <= S_MAGIC0;
                        end else if (length_with_byte == 0) begin
                            state <= S_CRC;
                        end else begin
                            state <= S_PAYLOAD;
                        end
                    end else begin
                        field_index <= field_index + 1'b1;
                    end
                end
                S_PAYLOAD: begin
                    if (command == 8'h01) begin
                        payload_we <= 1'b1;
                        payload_addr <= payload_index[ADDR_WIDTH-1:0];
                        payload_data <= rx_byte;
                    end
                    if (command == 8'h02 && payload_index < 4)
                        repeat_count <= repeat_count | ({24'd0, rx_byte} << (payload_index * 8));
                    crc_state <= crc32_byte(crc_state, rx_byte);
                    if (payload_index + 1 >= payload_len) begin
                        field_index <= 0;
                        received_crc <= 0;
                        state <= S_CRC;
                    end else begin
                        payload_index <= payload_index + 1'b1;
                    end
                end
                S_CRC: begin
                    received_crc <= received_crc_with_byte;
                    if (field_index == 3) begin
                        crc_ok <= (received_crc_with_byte == ~crc_state);
                        error_code <=
                            (received_crc_with_byte == ~crc_state) ? 8'd0 : 8'd3;
                        command_valid <= 1'b1;
                        field_index <= 0;
                        state <= S_MAGIC0;
                    end else begin
                        field_index <= field_index + 1'b1;
                    end
                end
                default: state <= S_MAGIC0;
            endcase
        end
    end
end

endmodule
