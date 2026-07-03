module reset_sync (
    input  wire clk,
    input  wire rst_n_async,
    output wire rst_n_sync
);

reg [1:0] sync_ff = 2'b00;

always @(posedge clk or negedge rst_n_async) begin
    if (!rst_n_async) begin
        sync_ff <= 2'b00;
    end else begin
        sync_ff <= {sync_ff[0], 1'b1};
    end
end

assign rst_n_sync = sync_ff[1];

endmodule
