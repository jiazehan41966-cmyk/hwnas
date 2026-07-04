module dynamic_validation_response (
    input  wire         clk,
    input  wire         rst,
    input  wire         start,
    input  wire [7:0]   status,
    input  wire [7:0]   command,
    input  wire [31:0]  sample_id,
    input  wire [31:0]  cycles,
    input  wire [63:0]  logits,
    input  wire [7:0]   argmax,
    input  wire [31:0]  checksum,
    input  wire [31:0]  repeat_count,
    output reg          byte_valid,
    input  wire         byte_ready,
    output reg [7:0]    byte_data,
    output reg          busy,
    output reg          done
);

reg [5:0] index;
reg [31:0] crc_state;
reg [7:0] latched_status;
reg [7:0] latched_command;
reg [31:0] latched_sample_id;
reg [31:0] latched_cycles;
reg [63:0] latched_logits;
reg [7:0] latched_argmax;
reg [31:0] latched_checksum;
reg [31:0] latched_repeat_count;
wire [31:0] crc_final = ~crc_state;

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

always @(*) begin
    case (index)
        0: byte_data = 8'h5A;
        1: byte_data = 8'hA5;
        2: byte_data = 8'd1;
        3: byte_data = latched_status;
        4: byte_data = latched_command;
        5,6,7,8: byte_data = latched_sample_id[(index-5)*8 +: 8];
        9,10,11,12: byte_data = latched_cycles[(index-9)*8 +: 8];
        13,14,15,16,17,18,19,20:
            byte_data = latched_logits[(index-13)*8 +: 8];
        21: byte_data = latched_argmax;
        22,23,24,25: byte_data = latched_checksum[(index-22)*8 +: 8];
        26,27,28,29: byte_data = latched_repeat_count[(index-26)*8 +: 8];
        30,31,32,33: byte_data = crc_final[(index-30)*8 +: 8];
        default: byte_data = 8'd0;
    endcase
end

always @(posedge clk) begin
    if (rst) begin
        index <= 0;
        crc_state <= 32'hFFFFFFFF;
        byte_valid <= 0;
        busy <= 0;
        done <= 0;
    end else begin
        done <= 0;
        if (start && !busy) begin
            latched_status <= status;
            latched_command <= command;
            latched_sample_id <= sample_id;
            latched_cycles <= cycles;
            latched_logits <= logits;
            latched_argmax <= argmax;
            latched_checksum <= checksum;
            latched_repeat_count <= repeat_count;
            index <= 0;
            crc_state <= 32'hFFFFFFFF;
            byte_valid <= 1'b1;
            busy <= 1'b1;
        end else if (busy && byte_valid && byte_ready) begin
            if (index >= 2 && index <= 29)
                crc_state <= crc32_byte(crc_state, byte_data);
            if (index == 33) begin
                byte_valid <= 0;
                busy <= 0;
                done <= 1'b1;
                index <= 0;
            end else begin
                index <= index + 1'b1;
            end
        end
    end
end

endmodule
