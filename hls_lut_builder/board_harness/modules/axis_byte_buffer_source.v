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

reg [31:0] word_index;
reg [ADDR_WIDTH-1:0] read_addr;
wire [7:0] memory_q;
reg [1:0] source_state;

localparam [1:0] S_IDLE = 2'd0;
localparam [1:0] S_READ_WAIT = 2'd1;
localparam [1:0] S_PRESENT = 2'd2;

// The deployment target is fixed to Xilinx 7-series. Explicit XPM prevents a
// 50,176-byte dynamic image from ever being expanded into LUT RAM/flip-flops.
xpm_memory_sdpram #(
    .ADDR_WIDTH_A(ADDR_WIDTH),
    .ADDR_WIDTH_B(ADDR_WIDTH),
    .AUTO_SLEEP_TIME(0),
    .BYTE_WRITE_WIDTH_A(8),
    .CLOCKING_MODE("common_clock"),
    .ECC_MODE("no_ecc"),
    .MEMORY_INIT_FILE("none"),
    .MEMORY_INIT_PARAM("0"),
    .MEMORY_OPTIMIZATION("true"),
    .MEMORY_PRIMITIVE("block"),
    .MEMORY_SIZE(BYTE_COUNT * 8),
    .MESSAGE_CONTROL(0),
    .READ_DATA_WIDTH_B(8),
    .READ_LATENCY_B(1),
    .READ_RESET_VALUE_B("0"),
    .RST_MODE_B("SYNC"),
    .SIM_ASSERT_CHK(0),
    .USE_EMBEDDED_CONSTRAINT(0),
    .USE_MEM_INIT(0),
    .WAKEUP_TIME("disable_sleep"),
    .WRITE_DATA_WIDTH_A(8),
    .WRITE_MODE_B("read_first")
) u_input_bram (
    .clka(clk),
    .ena(byte_we && byte_addr < BYTE_COUNT),
    .wea(byte_we && byte_addr < BYTE_COUNT),
    .addra(byte_addr),
    .dina(byte_data),
    .clkb(clk),
    .enb(1'b1),
    .addrb(read_addr),
    .doutb(memory_q),
    .rstb(!rst_n),
    .regceb(1'b1),
    .sleep(1'b0),
    .injectdbiterra(1'b0),
    .injectsbiterra(1'b0),
    .dbiterrb(),
    .sbiterrb()
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        word_index <= 0;
        read_addr <= 0;
        source_state <= S_IDLE;
        tdata <= {DATA_WIDTH{1'b0}};
        tkeep <= {KEEP_WIDTH{1'b0}};
        tstrb <= {KEEP_WIDTH{1'b0}};
        tlast <= 0;
        tvalid <= 0;
        done <= 0;
    end else begin
        if (start) begin
            word_index <= 0;
            read_addr <= 0;
            source_state <= S_READ_WAIT;
            tdata <= {DATA_WIDTH{1'b0}};
            tkeep <= {KEEP_WIDTH{1'b1}};
            tstrb <= {KEEP_WIDTH{1'b1}};
            tlast <= 1'b0;
            tvalid <= 1'b0;
            done <= 1'b0;
        end else if (source_state == S_READ_WAIT) begin
            // One explicit wait cycle keeps the BRAM read synchronous.
            source_state <= S_PRESENT;
        end else if (source_state == S_PRESENT && !tvalid) begin
            tdata <= memory_q;
            tlast <= (word_index + 1 >= WORD_COUNT);
            tvalid <= 1'b1;
        end else if (source_state == S_PRESENT && tvalid && tready) begin
            if (word_index + 1 >= WORD_COUNT) begin
                tvalid <= 1'b0;
                tlast <= 1'b0;
                done <= 1'b1;
                source_state <= S_IDLE;
            end else begin
                word_index <= word_index + 1'b1;
                read_addr <= word_index + 1'b1;
                tvalid <= 1'b0;
                tlast <= 1'b0;
                source_state <= S_READ_WAIT;
            end
        end
    end
end

endmodule
