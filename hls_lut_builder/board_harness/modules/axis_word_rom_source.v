module axis_word_rom_source #(
    parameter integer DATA_WIDTH = 8,
    parameter integer KEEP_WIDTH = (DATA_WIDTH / 8),
    parameter integer WORD_COUNT = 1,
    parameter WORD_MEM_FILE = ""
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    output reg [DATA_WIDTH-1:0]   tdata,
    output reg [KEEP_WIDTH-1:0]   tkeep,
    output reg [KEEP_WIDTH-1:0]   tstrb,
    output reg                    tlast,
    output reg                    tvalid,
    input  wire                   tready,
    output reg                    done
);

localparam integer ADDR_WIDTH = (WORD_COUNT <= 2) ? 1 : $clog2(WORD_COUNT);

(* rom_style = "block", ram_style = "block" *) reg [DATA_WIDTH-1:0] word_rom [0:WORD_COUNT-1];
reg [ADDR_WIDTH-1:0] index_reg = {ADDR_WIDTH{1'b0}};
reg [DATA_WIDTH-1:0] current_word = {DATA_WIDTH{1'b0}};
reg active = 1'b0;

integer i;
initial begin
    for (i = 0; i < WORD_COUNT; i = i + 1) begin
        word_rom[i] = {DATA_WIDTH{1'b0}};
    end
    if (WORD_MEM_FILE != "") begin
        $readmemh(WORD_MEM_FILE, word_rom);
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        index_reg <= {ADDR_WIDTH{1'b0}};
        current_word <= {DATA_WIDTH{1'b0}};
        active <= 1'b0;
        tdata <= {DATA_WIDTH{1'b0}};
        tkeep <= {KEEP_WIDTH{1'b1}};
        tstrb <= {KEEP_WIDTH{1'b1}};
        tlast <= 1'b0;
        tvalid <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        tkeep <= {KEEP_WIDTH{1'b1}};
        tstrb <= {KEEP_WIDTH{1'b1}};

        if (start) begin
            index_reg <= {ADDR_WIDTH{1'b0}};
            current_word <= word_rom[{ADDR_WIDTH{1'b0}}];
            active <= 1'b1;
            tlast <= 1'b0;
            tvalid <= 1'b0;
        end else if (active && !tvalid) begin
            tdata <= current_word;
            tlast <= (index_reg == WORD_COUNT - 1);
            tvalid <= 1'b1;
        end else if (active && tvalid && tready) begin
            if (index_reg == WORD_COUNT - 1) begin
                active <= 1'b0;
                tlast <= 1'b0;
                tvalid <= 1'b0;
                done <= 1'b1;
            end else begin
                index_reg <= index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1};
                current_word <= word_rom[index_reg + {{(ADDR_WIDTH-1){1'b0}}, 1'b1}];
                tlast <= 1'b0;
                tvalid <= 1'b0;
            end
        end
    end
end

endmodule
