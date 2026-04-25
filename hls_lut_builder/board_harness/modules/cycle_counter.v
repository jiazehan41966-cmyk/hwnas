module cycle_counter (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        clear,
    input  wire        start,
    input  wire        stop,
    output reg [31:0]  cycles,
    output reg         running,
    output reg         done
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        cycles <= 32'd0;
        running <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        if (clear) begin
            cycles <= 32'd0;
            running <= 1'b0;
        end else begin
            if (start) begin
                cycles <= 32'd0;
                running <= 1'b1;
            end else if (running) begin
                cycles <= cycles + 32'd1;
            end

            if (stop && running) begin
                running <= 1'b0;
                done <= 1'b1;
            end
        end
    end
end

endmodule
