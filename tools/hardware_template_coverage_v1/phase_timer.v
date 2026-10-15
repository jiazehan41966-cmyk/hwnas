`timescale 1ns / 1ps
// Same-clock observer of the HLS phase output. Results describe phase-marked
// intervals, not DDR bus occupancy. 1->2->3 may repeat across tiles.
// Results commit once at the final phase 0; illegal transitions fail closed.
module phase_timer (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK",
       X_INTERFACE_PARAMETER = "ASSOCIATED_RESET aresetn" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST",
       X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire aresetn,
    input wire [7:0] phase,
    input wire phase_valid,
    output reg [31:0] compute_cycles = 0,
    output reg [31:0] transfer_cycles = 0,
    output reg [31:0] total_cycles = 0,
    output reg [31:0] sequence_count = 0,
    output wire [31:0] identity,
    output wire [31:0] status
);
    localparam [31:0] MAX_COUNT = 32'hffffffff;
    reg [31:0] work_compute = 0, work_transfer = 0, work_total = 0;
    reg busy = 0, valid = 0;
    reg work_overflow = 0, work_error = 0;
    reg result_overflow = 0, result_error = 0, sequence_overflow = 0;
    reg [7:0] previous_phase = 0;
    reg [7:0] latched_phase = 0;
    wire [7:0] observed_phase = phase_valid ? phase : latched_phase;
    assign identity = 32'h50335431; // ASCII P3T1; repeated tiled load/compute/store ABI.
    // Bits 0 valid, 1 busy, 2 cycle overflow, 3 phase error, 4 sequence overflow.
    assign status = {27'b0, sequence_overflow, result_error, result_overflow, busy, valid};

    // ap_vld marks an actual C assignment. The HLS data output can return to
    // an old value when valid is low; that stale value must never restart us.
    always @(posedge clk) begin
        if (!aresetn) latched_phase <= 0;
        else if (phase_valid) latched_phase <= phase;
    end

    always @(posedge clk) begin
        if (!aresetn) begin
            compute_cycles <= 0;
            transfer_cycles <= 0;
            total_cycles <= 0;
            sequence_count <= 0;
            work_compute <= 0;
            work_transfer <= 0;
            work_total <= 0;
            busy <= 0;
            valid <= 0;
            work_overflow <= 0;
            work_error <= 0;
            result_overflow <= 0;
            result_error <= 0;
            sequence_overflow <= 0;
            previous_phase <= 0;
        end else if (!busy) begin
            if (observed_phase != 0) begin
                busy <= 1;
                valid <= 0;
                work_compute <= (observed_phase == 2) ? 1 : 0;
                work_transfer <= (observed_phase == 1 || observed_phase == 3) ? 1 : 0;
                work_total <= 1;
                work_overflow <= 0;
                work_error <= (observed_phase != 1);
                previous_phase <= observed_phase;
            end
        end else if (observed_phase == 0) begin
            compute_cycles <= work_compute;
            transfer_cycles <= work_transfer;
            total_cycles <= work_total;
            result_overflow <= work_overflow;
            result_error <= work_error || (previous_phase != 3);
            busy <= 0;
            valid <= !work_overflow && !work_error && (previous_phase == 3)
                     && !sequence_overflow && (sequence_count != MAX_COUNT);
            previous_phase <= 0;
            if (sequence_count == MAX_COUNT) begin
                sequence_overflow <= 1;
            end else begin
                sequence_count <= sequence_count + 1;
            end
        end else begin
            if (!((observed_phase == previous_phase && observed_phase <= 3) ||
                  (previous_phase == 1 && observed_phase == 2) ||
                  (previous_phase == 2 && observed_phase == 3) ||
                  (previous_phase == 3 && observed_phase == 1)))
                work_error <= 1;
            previous_phase <= observed_phase;
            if (work_total == MAX_COUNT) work_overflow <= 1;
            else work_total <= work_total + 1;
            if (observed_phase == 2) begin
                if (work_compute == MAX_COUNT) work_overflow <= 1;
                else work_compute <= work_compute + 1;
            end else if (observed_phase == 1 || observed_phase == 3) begin
                if (work_transfer == MAX_COUNT) work_overflow <= 1;
                else work_transfer <= work_transfer + 1;
            end
        end
    end
endmodule
