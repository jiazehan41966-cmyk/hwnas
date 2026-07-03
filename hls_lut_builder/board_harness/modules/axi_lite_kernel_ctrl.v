module axi_lite_kernel_ctrl #(
    parameter integer ADDR_WIDTH = 4,
    parameter integer DATA_WIDTH = 32
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    output reg                    done,
    output reg                    busy,
    output reg                    error,
    output reg                    m_axi_awvalid,
    input  wire                   m_axi_awready,
    output reg [ADDR_WIDTH-1:0]   m_axi_awaddr,
    output reg                    m_axi_wvalid,
    input  wire                   m_axi_wready,
    output reg [DATA_WIDTH-1:0]   m_axi_wdata,
    output reg [(DATA_WIDTH/8)-1:0] m_axi_wstrb,
    input  wire                   m_axi_bvalid,
    output reg                    m_axi_bready,
    input  wire [1:0]             m_axi_bresp,
    output reg                    m_axi_arvalid,
    input  wire                   m_axi_arready,
    output reg [ADDR_WIDTH-1:0]   m_axi_araddr,
    input  wire                   m_axi_rvalid,
    output reg                    m_axi_rready,
    input  wire [DATA_WIDTH-1:0]  m_axi_rdata,
    input  wire [1:0]             m_axi_rresp
);

localparam [3:0]
    ST_IDLE       = 4'd0,
    ST_WRITE_REQ  = 4'd1,
    ST_WRITE_RESP = 4'd2,
    ST_POLL_ADDR  = 4'd3,
    ST_POLL_DATA  = 4'd4,
    ST_DONE       = 4'd5,
    ST_ERROR      = 4'd6;

reg [3:0] state = ST_IDLE;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= ST_IDLE;
        done <= 1'b0;
        busy <= 1'b0;
        error <= 1'b0;
        m_axi_awvalid <= 1'b0;
        m_axi_awaddr <= {ADDR_WIDTH{1'b0}};
        m_axi_wvalid <= 1'b0;
        m_axi_wdata <= {DATA_WIDTH{1'b0}};
        m_axi_wstrb <= {(DATA_WIDTH/8){1'b1}};
        m_axi_bready <= 1'b0;
        m_axi_arvalid <= 1'b0;
        m_axi_araddr <= {ADDR_WIDTH{1'b0}};
        m_axi_rready <= 1'b0;
    end else begin
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                error <= 1'b0;
                m_axi_awvalid <= 1'b0;
                m_axi_wvalid <= 1'b0;
                m_axi_bready <= 1'b0;
                m_axi_arvalid <= 1'b0;
                m_axi_rready <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    m_axi_awaddr <= {ADDR_WIDTH{1'b0}};
                    m_axi_wdata <= {{(DATA_WIDTH-1){1'b0}}, 1'b1};
                    m_axi_awvalid <= 1'b1;
                    m_axi_wvalid <= 1'b1;
                    state <= ST_WRITE_REQ;
                end
            end

            ST_WRITE_REQ: begin
                if (m_axi_awvalid && m_axi_awready) begin
                    m_axi_awvalid <= 1'b0;
                end
                if (m_axi_wvalid && m_axi_wready) begin
                    m_axi_wvalid <= 1'b0;
                end
                if ((!m_axi_awvalid || m_axi_awready) && (!m_axi_wvalid || m_axi_wready)) begin
                    m_axi_bready <= 1'b1;
                    state <= ST_WRITE_RESP;
                end
            end

            ST_WRITE_RESP: begin
                if (m_axi_bvalid) begin
                    m_axi_bready <= 1'b0;
                    if (m_axi_bresp != 2'b00) begin
                        error <= 1'b1;
                        state <= ST_ERROR;
                    end else begin
                        m_axi_araddr <= {ADDR_WIDTH{1'b0}};
                        m_axi_arvalid <= 1'b1;
                        state <= ST_POLL_ADDR;
                    end
                end
            end

            ST_POLL_ADDR: begin
                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;
                    state <= ST_POLL_DATA;
                end
            end

            ST_POLL_DATA: begin
                if (m_axi_rvalid) begin
                    m_axi_rready <= 1'b0;
                    if (m_axi_rresp != 2'b00) begin
                        error <= 1'b1;
                        state <= ST_ERROR;
                    end else if (m_axi_rdata[1] || m_axi_rdata[3]) begin
                        state <= ST_DONE;
                    end else begin
                        m_axi_araddr <= {ADDR_WIDTH{1'b0}};
                        m_axi_arvalid <= 1'b1;
                        state <= ST_POLL_ADDR;
                    end
                end
            end

            ST_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= ST_IDLE;
            end

            default: begin
                busy <= 1'b0;
                state <= ST_IDLE;
            end
        endcase

        if (state == ST_ERROR) begin
            busy <= 1'b0;
            done <= 1'b1;
            m_axi_awvalid <= 1'b0;
            m_axi_wvalid <= 1'b0;
            m_axi_bready <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            state <= ST_IDLE;
        end
    end
end

endmodule
