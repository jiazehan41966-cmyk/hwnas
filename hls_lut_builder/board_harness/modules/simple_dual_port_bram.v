module simple_dual_port_bram #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ADDR_WIDTH = 8,
    parameter integer DEPTH = 256,
    parameter INIT_FILE = ""
) (
    input  wire                         clka,
    input  wire                         rsta,
    input  wire                         ena,
    input  wire [(DATA_WIDTH/8)-1:0]    wea,
    input  wire [ADDR_WIDTH-1:0]        addra,
    input  wire [DATA_WIDTH-1:0]        dina,
    output reg  [DATA_WIDTH-1:0]        douta,
    input  wire                         clkb,
    input  wire                         rstb,
    input  wire                         enb,
    input  wire [(DATA_WIDTH/8)-1:0]    web,
    input  wire [ADDR_WIDTH-1:0]        addrb,
    input  wire [DATA_WIDTH-1:0]        dinb,
    output reg  [DATA_WIDTH-1:0]        doutb
);

reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
integer i;
integer byte_idx;

initial begin
    for (i = 0; i < DEPTH; i = i + 1) begin
        mem[i] = {DATA_WIDTH{1'b0}};
    end
    if (INIT_FILE != "") begin
        $readmemh(INIT_FILE, mem);
    end
end

always @(posedge clka) begin
    if (rsta) begin
        douta <= {DATA_WIDTH{1'b0}};
    end else if (ena) begin
        if (addra < DEPTH) begin
            for (byte_idx = 0; byte_idx < (DATA_WIDTH/8); byte_idx = byte_idx + 1) begin
                if (wea[byte_idx]) begin
                    mem[addra][byte_idx*8 +: 8] <= dina[byte_idx*8 +: 8];
                end
            end
            douta <= mem[addra];
        end else begin
            douta <= {DATA_WIDTH{1'b0}};
        end
    end
end

always @(posedge clkb) begin
    if (rstb) begin
        doutb <= {DATA_WIDTH{1'b0}};
    end else if (enb) begin
        if (addrb < DEPTH) begin
            for (byte_idx = 0; byte_idx < (DATA_WIDTH/8); byte_idx = byte_idx + 1) begin
                if (web[byte_idx]) begin
                    mem[addrb][byte_idx*8 +: 8] <= dinb[byte_idx*8 +: 8];
                end
            end
            doutb <= mem[addrb];
        end else begin
            doutb <= {DATA_WIDTH{1'b0}};
        end
    end
end

endmodule
