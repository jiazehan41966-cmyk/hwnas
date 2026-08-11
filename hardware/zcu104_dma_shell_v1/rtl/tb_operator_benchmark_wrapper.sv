`timescale 1ns / 1ps

module tb_operator_benchmark_wrapper;
    localparam int WORDS = 784;

    reg aclk = 1'b0;
    reg aresetn = 1'b0;
    always #2.5 aclk = ~aclk;

    reg [7:0] awaddr = 0;
    reg [2:0] awprot = 0;
    reg awvalid = 0;
    wire awready;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    reg wvalid = 0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 0;
    reg [7:0] araddr = 0;
    reg [2:0] arprot = 0;
    reg arvalid = 0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 0;

    reg [255:0] dma_in_data = 0;
    reg [31:0] dma_in_keep = 0;
    reg [31:0] dma_in_strb = 0;
    reg dma_in_last = 0;
    reg dma_in_valid = 0;
    wire dma_in_ready;

    wire [255:0] dma_out_data;
    wire [31:0] dma_out_keep;
    wire [31:0] dma_out_strb;
    wire dma_out_last;
    wire dma_out_valid;
    reg dma_out_ready = 1'b1;

    wire [255:0] core_in_data;
    wire [31:0] core_in_keep;
    wire [31:0] core_in_strb;
    wire core_in_last;
    wire core_in_valid;
    wire core_in_ready;

`ifdef ACTUAL_HLS_CORE
    wire [255:0] core_out_data;
    wire [31:0] core_out_keep;
    wire [31:0] core_out_strb;
    wire core_out_last;
    wire core_out_valid;
`else
    reg [255:0] core_out_data = 0;
    reg [31:0] core_out_keep = 0;
    reg [31:0] core_out_strb = 0;
    reg core_out_last = 0;
    reg core_out_valid = 0;
`endif
    wire core_out_ready;

    wire core_ap_start;
`ifdef ACTUAL_HLS_CORE
    wire core_ap_done;
    wire core_ap_idle;
    wire core_ap_ready;
`else
    reg core_ap_done = 0;
    reg core_ap_idle = 1;
    reg core_ap_ready = 1;
`endif
    wire irq;

    integer outputs_seen = 0;
    integer errors = 0;
    reg [31:0] rd_value;
    reg [31:0] measured_core_cycles = 0;

`ifndef ACTUAL_HLS_CORE
    assign core_in_ready = !core_out_valid || core_out_ready;
`endif

    operator_benchmark_wrapper #(
        .INPUT_WORDS(WORDS),
        .OUTPUT_WORDS(WORDS)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axi_ctrl_awaddr(awaddr),
        .s_axi_ctrl_awprot(awprot),
        .s_axi_ctrl_awvalid(awvalid),
        .s_axi_ctrl_awready(awready),
        .s_axi_ctrl_wdata(wdata),
        .s_axi_ctrl_wstrb(wstrb),
        .s_axi_ctrl_wvalid(wvalid),
        .s_axi_ctrl_wready(wready),
        .s_axi_ctrl_bresp(bresp),
        .s_axi_ctrl_bvalid(bvalid),
        .s_axi_ctrl_bready(bready),
        .s_axi_ctrl_araddr(araddr),
        .s_axi_ctrl_arprot(arprot),
        .s_axi_ctrl_arvalid(arvalid),
        .s_axi_ctrl_arready(arready),
        .s_axi_ctrl_rdata(rdata),
        .s_axi_ctrl_rresp(rresp),
        .s_axi_ctrl_rvalid(rvalid),
        .s_axi_ctrl_rready(rready),
        .s_axis_dma_tdata(dma_in_data),
        .s_axis_dma_tkeep(dma_in_keep),
        .s_axis_dma_tstrb(dma_in_strb),
        .s_axis_dma_tlast(dma_in_last),
        .s_axis_dma_tvalid(dma_in_valid),
        .s_axis_dma_tready(dma_in_ready),
        .m_axis_dma_tdata(dma_out_data),
        .m_axis_dma_tkeep(dma_out_keep),
        .m_axis_dma_tstrb(dma_out_strb),
        .m_axis_dma_tlast(dma_out_last),
        .m_axis_dma_tvalid(dma_out_valid),
        .m_axis_dma_tready(dma_out_ready),
        .m_axis_core_tdata(core_in_data),
        .m_axis_core_tkeep(core_in_keep),
        .m_axis_core_tstrb(core_in_strb),
        .m_axis_core_tlast(core_in_last),
        .m_axis_core_tvalid(core_in_valid),
        .m_axis_core_tready(core_in_ready),
        .s_axis_core_tdata(core_out_data),
        .s_axis_core_tkeep(core_out_keep),
        .s_axis_core_tstrb(core_out_strb),
        .s_axis_core_tlast(core_out_last),
        .s_axis_core_tvalid(core_out_valid),
        .s_axis_core_tready(core_out_ready),
        .core_ap_start(core_ap_start),
        .core_ap_done(core_ap_done),
        .core_ap_idle(core_ap_idle),
        .core_ap_ready(core_ap_ready),
        .irq(irq)
    );

`ifdef ACTUAL_HLS_CORE
    zcu104_stage4_skip_core actual_skip_core (
        .ap_clk(aclk),
        .ap_rst_n(aresetn),
        .ap_start(core_ap_start),
        .ap_done(core_ap_done),
        .ap_idle(core_ap_idle),
        .ap_ready(core_ap_ready),
        .input_stream_TVALID(core_in_valid),
        .output_stream_TREADY(core_out_ready),
        .input_stream_TDATA(core_in_data),
        .input_stream_TREADY(core_in_ready),
        .input_stream_TKEEP(core_in_keep),
        .input_stream_TSTRB(core_in_strb),
        .input_stream_TLAST(core_in_last),
        .output_stream_TDATA(core_out_data),
        .output_stream_TVALID(core_out_valid),
        .output_stream_TKEEP(core_out_keep),
        .output_stream_TSTRB(core_out_strb),
        .output_stream_TLAST(core_out_last)
    );
`else
    always @(posedge aclk) begin
        if (!aresetn) begin
            core_out_valid <= 1'b0;
            core_out_data <= 0;
            core_out_keep <= 0;
            core_out_strb <= 0;
            core_out_last <= 0;
            core_ap_idle <= 1'b1;
            core_ap_done <= 1'b0;
        end else begin
            core_ap_done <= 1'b0;
            if (core_ap_start) begin
                core_ap_idle <= 1'b0;
            end
            if (core_in_ready) begin
                core_out_valid <= core_in_valid;
                if (core_in_valid) begin
                    core_out_data <= core_in_data;
                    core_out_keep <= core_in_keep;
                    core_out_strb <= core_in_strb;
                    core_out_last <= core_in_last;
                end
            end
            if (core_out_valid && core_out_ready && core_out_last) begin
                core_ap_done <= 1'b1;
                core_ap_idle <= 1'b1;
            end
        end
    end
`endif

    always @(posedge aclk) begin
        if (aresetn && dma_out_valid && dma_out_ready) begin
            if (dma_out_data[31:0] !== outputs_seen[31:0]) begin
                errors = errors + 1;
            end
            if (dma_out_keep !== 32'hffff_ffff || dma_out_strb !== 32'hffff_ffff) begin
                errors = errors + 1;
            end
            if (dma_out_last !== (outputs_seen == WORDS - 1)) begin
                errors = errors + 1;
            end
            outputs_seen = outputs_seen + 1;
        end
    end

    task automatic axil_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            awaddr <= addr;
            awvalid <= 1'b1;
            while (!awready) @(posedge aclk);
            @(posedge aclk);
            awvalid <= 1'b0;

            // Deliberately issue W after AW to verify independent channels.
            @(posedge aclk);
            wdata <= data;
            wstrb <= 4'hf;
            wvalid <= 1'b1;
            while (!wready) @(posedge aclk);
            @(posedge aclk);
            wvalid <= 1'b0;
            bready <= 1'b1;
            while (!bvalid) @(posedge aclk);
            @(posedge aclk);
            bready <= 1'b0;
        end
    endtask

    task automatic axil_read(input [7:0] addr, output [31:0] data);
        begin
            @(posedge aclk);
            araddr <= addr;
            arvalid <= 1'b1;
            while (!arready) @(posedge aclk);
            @(posedge aclk);
            arvalid <= 1'b0;
            rready <= 1'b1;
            while (!rvalid) @(posedge aclk);
            data = rdata;
            @(posedge aclk);
            rready <= 1'b0;
        end
    endtask

    initial begin
        repeat (8) @(posedge aclk);
        aresetn <= 1'b1;
        repeat (4) @(posedge aclk);

        axil_write(8'h00, 32'h0000_0001);

        for (int word = 0; word < WORDS; word++) begin
            dma_in_data <= 0;
            dma_in_data[31:0] <= word;
            dma_in_keep <= 32'hffff_ffff;
            dma_in_strb <= 32'hffff_ffff;
            dma_in_last <= (word == WORDS - 1);
            dma_in_valid <= 1'b1;
            do @(posedge aclk); while (!dma_in_ready);
        end
        dma_in_valid <= 1'b0;
        dma_in_last <= 1'b0;

        wait (irq === 1'b1);
        repeat (3) @(posedge aclk);

        axil_read(8'h04, rd_value);
        if (rd_value[2:0] !== 3'b101 || rd_value[4] !== 1'b1) errors = errors + 1;
        axil_read(8'h08, rd_value);
        if (rd_value !== 32'd1) errors = errors + 1;
        axil_read(8'h0c, rd_value);
        if (rd_value !== 32'd0) errors = errors + 1;
        axil_read(8'h10, rd_value);
        measured_core_cycles = rd_value;
`ifdef ACTUAL_HLS_CORE
        if (rd_value < WORDS || rd_value > (WORDS + 32)) begin
            $display("unexpected actual HLS core cycle count: %0d", rd_value);
            errors = errors + 1;
        end
`else
        if (rd_value !== 32'd785) begin
            $display("unexpected core cycle count: %0d", rd_value);
            errors = errors + 1;
        end
`endif
        axil_read(8'h20, rd_value);
        if (rd_value !== WORDS) errors = errors + 1;
        axil_read(8'h24, rd_value);
        if (rd_value !== WORDS) errors = errors + 1;
        axil_read(8'h28, rd_value);
        if (rd_value !== 32'd0) errors = errors + 1;
        axil_read(8'h2c, rd_value);
        if (rd_value !== 32'h444d_4153) errors = errors + 1;
        if (outputs_seen !== WORDS) errors = errors + 1;

        if (errors == 0) begin
`ifdef ACTUAL_HLS_CORE
            $display("PASS: wrapper plus actual exported HLS Skip RTL, 784-word identity, TLAST, registers, acknowledged ap_start, %0d core cycles", measured_core_cycles);
`else
            $display("PASS: wrapper transaction, register map, TLAST and 785-cycle handshake latency");
`endif
            $finish;
        end else begin
            $fatal(1, "wrapper simulation failed with %0d errors", errors);
        end
    end

    initial begin
        #2ms;
        $fatal(1, "simulation timeout");
    end
endmodule
