`timescale 1ns / 1ps

module operator_benchmark_wrapper #(
    parameter integer AXIS_DATA_BITS = 256,
    parameter integer INPUT_WORDS = 784,
    parameter integer OUTPUT_WORDS = 784,
    parameter [31:0] OPERATOR_ID = 32'h0004_0001,
    parameter [31:0] CLOCK_HZ = 32'd200000000,
    parameter integer AXIL_ADDR_BITS = 8
) (
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ACLK, ASSOCIATED_BUSIF S_AXI_CTRL:S_AXIS_DMA:M_AXIS_DMA:M_AXIS_CORE:S_AXIS_CORE, ASSOCIATED_RESET ARESETN, FREQ_HZ 200000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ACLK CLK" *)
    input  wire                         aclk,
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME ARESETN, POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ARESETN RST" *)
    input  wire                         aresetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWADDR" *)
    input  wire [AXIL_ADDR_BITS-1:0]    s_axi_ctrl_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWPROT" *)
    input  wire [2:0]                   s_axi_ctrl_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWVALID" *)
    input  wire                         s_axi_ctrl_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL AWREADY" *)
    output wire                         s_axi_ctrl_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WDATA" *)
    input  wire [31:0]                  s_axi_ctrl_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WSTRB" *)
    input  wire [3:0]                   s_axi_ctrl_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WVALID" *)
    input  wire                         s_axi_ctrl_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL WREADY" *)
    output wire                         s_axi_ctrl_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BRESP" *)
    output wire [1:0]                   s_axi_ctrl_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BVALID" *)
    output wire                         s_axi_ctrl_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL BREADY" *)
    input  wire                         s_axi_ctrl_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARADDR" *)
    input  wire [AXIL_ADDR_BITS-1:0]    s_axi_ctrl_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARPROT" *)
    input  wire [2:0]                   s_axi_ctrl_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARVALID" *)
    input  wire                         s_axi_ctrl_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL ARREADY" *)
    output wire                         s_axi_ctrl_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RDATA" *)
    output wire [31:0]                  s_axi_ctrl_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RRESP" *)
    output wire [1:0]                   s_axi_ctrl_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RVALID" *)
    output wire                         s_axi_ctrl_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CTRL RREADY" *)
    input  wire                         s_axi_ctrl_rready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS_DMA, TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TSTRB 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TDATA" *)
    input  wire [AXIS_DATA_BITS-1:0]    s_axis_dma_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TKEEP" *)
    input  wire [(AXIS_DATA_BITS/8)-1:0] s_axis_dma_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TSTRB" *)
    input  wire [(AXIS_DATA_BITS/8)-1:0] s_axis_dma_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TLAST" *)
    input  wire                         s_axis_dma_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TVALID" *)
    input  wire                         s_axis_dma_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_DMA TREADY" *)
    output wire                         s_axis_dma_tready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS_DMA, TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TSTRB 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TDATA" *)
    output wire [AXIS_DATA_BITS-1:0]    m_axis_dma_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TKEEP" *)
    output wire [(AXIS_DATA_BITS/8)-1:0] m_axis_dma_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TSTRB" *)
    output wire [(AXIS_DATA_BITS/8)-1:0] m_axis_dma_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TLAST" *)
    output wire                         m_axis_dma_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TVALID" *)
    output wire                         m_axis_dma_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_DMA TREADY" *)
    input  wire                         m_axis_dma_tready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS_CORE, TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TSTRB 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TDATA" *)
    output wire [AXIS_DATA_BITS-1:0]    m_axis_core_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TKEEP" *)
    output wire [(AXIS_DATA_BITS/8)-1:0] m_axis_core_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TSTRB" *)
    output wire [(AXIS_DATA_BITS/8)-1:0] m_axis_core_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TLAST" *)
    output wire                         m_axis_core_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TVALID" *)
    output wire                         m_axis_core_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_CORE TREADY" *)
    input  wire                         m_axis_core_tready,

    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS_CORE, TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TSTRB 1, HAS_TLAST 1" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TDATA" *)
    input  wire [AXIS_DATA_BITS-1:0]    s_axis_core_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TKEEP" *)
    input  wire [(AXIS_DATA_BITS/8)-1:0] s_axis_core_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TSTRB" *)
    input  wire [(AXIS_DATA_BITS/8)-1:0] s_axis_core_tstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TLAST" *)
    input  wire                         s_axis_core_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TVALID" *)
    input  wire                         s_axis_core_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_CORE TREADY" *)
    output wire                         s_axis_core_tready,

    output reg                          core_ap_start,
    input  wire                         core_ap_done,
    input  wire                         core_ap_idle,
    input  wire                         core_ap_ready,
    output wire                         irq
);

    localparam integer AXIS_BYTES = AXIS_DATA_BITS / 8;
    localparam [AXIS_BYTES-1:0] FULL_BYTE_MASK = {AXIS_BYTES{1'b1}};

    localparam [7:0] REG_CONTROL          = 8'h00;
    localparam [7:0] REG_STATUS           = 8'h04;
    localparam [7:0] REG_EXEC_COUNT_LO    = 8'h08;
    localparam [7:0] REG_EXEC_COUNT_HI    = 8'h0c;
    localparam [7:0] REG_CORE_CYCLES_LO   = 8'h10;
    localparam [7:0] REG_CORE_CYCLES_HI   = 8'h14;
    localparam [7:0] REG_INPUT_WORDS      = 8'h18;
    localparam [7:0] REG_OUTPUT_WORDS     = 8'h1c;
    localparam [7:0] REG_INPUT_SEEN       = 8'h20;
    localparam [7:0] REG_OUTPUT_SEEN      = 8'h24;
    localparam [7:0] REG_ERROR_FLAGS      = 8'h28;
    localparam [7:0] REG_ID               = 8'h2c;
    localparam [7:0] REG_VERSION          = 8'h30;
    localparam [7:0] REG_OPERATOR_ID      = 8'h34;
    localparam [7:0] REG_AXIS_DATA_BITS   = 8'h38;
    localparam [7:0] REG_CLOCK_HZ         = 8'h3c;

    reg [AXIL_ADDR_BITS-1:0] awaddr_hold;
    reg [31:0] wdata_hold;
    reg [3:0] wstrb_hold;
    reg aw_hold_valid;
    reg w_hold_valid;
    reg bvalid_reg;
    reg rvalid_reg;
    reg [31:0] rdata_reg;

    reg busy;
    reg done_sticky;
    reg core_done_seen;
    reg input_started;
    reg output_started;
    reg [63:0] execution_count;
    reg [63:0] core_latency_cycles;
    reg [63:0] core_cycles_live;
    reg core_timing_active;
    reg [31:0] input_words_seen;
    reg [31:0] output_words_seen;
    reg [31:0] error_flags;

    wire write_commit = aw_hold_valid && w_hold_valid && !bvalid_reg;
    wire [7:0] write_addr = awaddr_hold[7:0];
    wire start_request = write_commit && (write_addr == REG_CONTROL) && wstrb_hold[0] && wdata_hold[0];
    wire clear_done_request = write_commit && (write_addr == REG_CONTROL) && wstrb_hold[0] && wdata_hold[1];
    wire clear_errors_request = write_commit && (write_addr == REG_CONTROL) && wstrb_hold[0] && wdata_hold[2];

    wire input_final_expected = input_words_seen == (INPUT_WORDS - 1);
    wire output_final_expected = output_words_seen == (OUTPUT_WORDS - 1);

    assign s_axi_ctrl_awready = !aw_hold_valid;
    assign s_axi_ctrl_wready = !w_hold_valid;
    assign s_axi_ctrl_bvalid = bvalid_reg;
    assign s_axi_ctrl_bresp = 2'b00;
    assign s_axi_ctrl_arready = !rvalid_reg;
    assign s_axi_ctrl_rvalid = rvalid_reg;
    assign s_axi_ctrl_rdata = rdata_reg;
    assign s_axi_ctrl_rresp = 2'b00;

    assign m_axis_core_tdata = s_axis_dma_tdata;
    assign m_axis_core_tkeep = FULL_BYTE_MASK;
    assign m_axis_core_tstrb = FULL_BYTE_MASK;
    assign m_axis_core_tlast = input_final_expected;
    assign m_axis_core_tvalid = busy && (input_words_seen < INPUT_WORDS) && s_axis_dma_tvalid;
    assign s_axis_dma_tready = busy && (input_words_seen < INPUT_WORDS) && m_axis_core_tready;

    assign m_axis_dma_tdata = s_axis_core_tdata;
    assign m_axis_dma_tkeep = FULL_BYTE_MASK;
    assign m_axis_dma_tstrb = FULL_BYTE_MASK;
    assign m_axis_dma_tlast = output_final_expected;
    assign m_axis_dma_tvalid = busy && (output_words_seen < OUTPUT_WORDS) && s_axis_core_tvalid;
    assign s_axis_core_tready = busy && (output_words_seen < OUTPUT_WORDS) && m_axis_dma_tready;

    wire input_handshake = m_axis_core_tvalid && m_axis_core_tready;
    wire output_handshake = s_axis_core_tvalid && s_axis_core_tready;
    wire final_output_handshake = output_handshake && output_final_expected;

    assign irq = done_sticky;

    function [31:0] read_register;
        input [7:0] addr;
        begin
            case (addr)
                REG_CONTROL:        read_register = 32'h0000_0000;
                REG_STATUS:         read_register = {
                    23'd0,
                    output_started,
                    input_started,
                    core_done_seen,
                    core_ap_ready,
                    core_ap_idle,
                    (|error_flags),
                    done_sticky,
                    busy,
                    !busy
                };
                REG_EXEC_COUNT_LO:  read_register = execution_count[31:0];
                REG_EXEC_COUNT_HI:  read_register = execution_count[63:32];
                REG_CORE_CYCLES_LO: read_register = core_latency_cycles[31:0];
                REG_CORE_CYCLES_HI: read_register = core_latency_cycles[63:32];
                REG_INPUT_WORDS:    read_register = INPUT_WORDS;
                REG_OUTPUT_WORDS:   read_register = OUTPUT_WORDS;
                REG_INPUT_SEEN:     read_register = input_words_seen;
                REG_OUTPUT_SEEN:    read_register = output_words_seen;
                REG_ERROR_FLAGS:    read_register = error_flags;
                REG_ID:             read_register = 32'h444d_4153;
                REG_VERSION:        read_register = 32'h0001_0000;
                REG_OPERATOR_ID:    read_register = OPERATOR_ID;
                REG_AXIS_DATA_BITS: read_register = AXIS_DATA_BITS;
                REG_CLOCK_HZ:       read_register = CLOCK_HZ;
                default:            read_register = 32'h0000_0000;
            endcase
        end
    endfunction

    always @(posedge aclk) begin
        if (!aresetn) begin
            awaddr_hold <= {AXIL_ADDR_BITS{1'b0}};
            wdata_hold <= 32'd0;
            wstrb_hold <= 4'd0;
            aw_hold_valid <= 1'b0;
            w_hold_valid <= 1'b0;
            bvalid_reg <= 1'b0;
            rvalid_reg <= 1'b0;
            rdata_reg <= 32'd0;
        end else begin
            if (s_axi_ctrl_awready && s_axi_ctrl_awvalid) begin
                awaddr_hold <= s_axi_ctrl_awaddr;
                aw_hold_valid <= 1'b1;
            end
            if (s_axi_ctrl_wready && s_axi_ctrl_wvalid) begin
                wdata_hold <= s_axi_ctrl_wdata;
                wstrb_hold <= s_axi_ctrl_wstrb;
                w_hold_valid <= 1'b1;
            end
            if (write_commit) begin
                aw_hold_valid <= 1'b0;
                w_hold_valid <= 1'b0;
                bvalid_reg <= 1'b1;
            end else if (bvalid_reg && s_axi_ctrl_bready) begin
                bvalid_reg <= 1'b0;
            end

            if (s_axi_ctrl_arready && s_axi_ctrl_arvalid) begin
                rdata_reg <= read_register(s_axi_ctrl_araddr[7:0]);
                rvalid_reg <= 1'b1;
            end else if (rvalid_reg && s_axi_ctrl_rready) begin
                rvalid_reg <= 1'b0;
            end
        end
    end

    always @(posedge aclk) begin
        if (!aresetn) begin
            busy <= 1'b0;
            done_sticky <= 1'b0;
            core_done_seen <= 1'b0;
            input_started <= 1'b0;
            output_started <= 1'b0;
            execution_count <= 64'd0;
            core_latency_cycles <= 64'd0;
            core_cycles_live <= 64'd0;
            core_timing_active <= 1'b0;
            input_words_seen <= 32'd0;
            output_words_seen <= 32'd0;
            error_flags <= 32'd0;
            core_ap_start <= 1'b0;
        end else begin
            if (clear_done_request) begin
                done_sticky <= 1'b0;
            end
            if (clear_errors_request) begin
                error_flags <= 32'd0;
            end

            if (start_request) begin
                if (busy) begin
                    error_flags[0] <= 1'b1;
                end else begin
                    busy <= 1'b1;
                    done_sticky <= 1'b0;
                    core_done_seen <= 1'b0;
                    input_started <= 1'b0;
                    output_started <= 1'b0;
                    core_latency_cycles <= 64'd0;
                    core_cycles_live <= 64'd0;
                    core_timing_active <= 1'b0;
                    input_words_seen <= 32'd0;
                    output_words_seen <= 32'd0;
                    error_flags <= 32'd0;
                    core_ap_start <= 1'b1;
                end
            end else if (core_ap_start && core_ap_ready) begin
                // ap_ctrl_hs requires ap_start to remain asserted until the
                // core acknowledges this request with ap_ready.
                core_ap_start <= 1'b0;
            end

            if (core_ap_done) begin
                core_done_seen <= 1'b1;
            end

            if (input_handshake) begin
                input_started <= 1'b1;
                input_words_seen <= input_words_seen + 1'b1;
                if (s_axis_dma_tlast && !input_final_expected) begin
                    error_flags[1] <= 1'b1;
                end
                if (!s_axis_dma_tlast && input_final_expected) begin
                    error_flags[2] <= 1'b1;
                end
                if (s_axis_dma_tkeep != FULL_BYTE_MASK) begin
                    error_flags[3] <= 1'b1;
                end
                if (s_axis_dma_tstrb != FULL_BYTE_MASK) begin
                    error_flags[4] <= 1'b1;
                end
            end

            if (output_handshake) begin
                output_started <= 1'b1;
                output_words_seen <= output_words_seen + 1'b1;
                if (s_axis_core_tlast && !output_final_expected) begin
                    error_flags[5] <= 1'b1;
                end
                if (!s_axis_core_tlast && output_final_expected) begin
                    error_flags[6] <= 1'b1;
                end
                if (s_axis_core_tkeep != FULL_BYTE_MASK) begin
                    error_flags[7] <= 1'b1;
                end
                if (s_axis_core_tstrb != FULL_BYTE_MASK) begin
                    error_flags[8] <= 1'b1;
                end
            end

            if (!core_timing_active && input_handshake) begin
                core_timing_active <= 1'b1;
                core_cycles_live <= 64'd1;
                if (final_output_handshake) begin
                    core_latency_cycles <= 64'd1;
                    core_timing_active <= 1'b0;
                end
            end else if (core_timing_active) begin
                if (final_output_handshake) begin
                    core_latency_cycles <= core_cycles_live + 1'b1;
                    core_timing_active <= 1'b0;
                end else begin
                    core_cycles_live <= core_cycles_live + 1'b1;
                end
            end

            if (final_output_handshake) begin
                busy <= 1'b0;
                done_sticky <= 1'b1;
                execution_count <= execution_count + 1'b1;
            end
        end
    end

    wire unused = &{1'b0, s_axi_ctrl_awprot, s_axi_ctrl_arprot};

endmodule
