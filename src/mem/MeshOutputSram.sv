module MeshOutputSram #(
    parameter DEPTH = 1024,
    parameter DATA_WIDTH = 32,
    parameter NUM_PORTS = 1,
    parameter WIDE = 1  // words per wide read
) (
    input logic clk_i,
    input logic rstn_i,

    input logic [NUM_PORTS-1:0]                 we_i,
    input logic [NUM_PORTS-1:0][          31:0] waddr_i,
    input logic [NUM_PORTS-1:0][DATA_WIDTH-1:0] wdata_i,

    input  logic                            wide_enable_i,
    input  logic [WIDE-1:0][          31:0] wide_addr_i,
    output logic [WIDE-1:0][DATA_WIDTH-1:0] wide_data_o,
    output logic                            wide_valid_o
);

  // The array is a mem_array (one module boundary, so synthesis can swap it for a macro): a write port per reducer, a read port per wide word.
  // No write-enable from reset: the reducers write nothing while in reset, and a reset net should reach only resets.
  localparam int AW = (DEPTH > 1) ? $clog2(DEPTH) : 1;
  logic [NUM_PORTS-1:0] we;
  logic [NUM_PORTS-1:0][AW-1:0] waddr;
  logic [WIDE-1:0][AW-1:0] raddr;
  logic [WIDE-1:0][DATA_WIDTH-1:0] rdata;
  logic [WIDE-1:0] in_range;  // per wide word: the last enabled read's address was inside the array (a word outside reads as 0)
  always_comb begin
    for (int p = 0; p < NUM_PORTS; p++) begin
      we[p]    = we_i[p] && waddr_i[p] < DEPTH;
      waddr[p] = AW'(waddr_i[p]);
    end
    for (int k = 0; k < WIDE; k++) begin
      raddr[k] = AW'(wide_addr_i[k]);
      wide_data_o[k] = in_range[k] ? rdata[k] : '0;
    end
  end
  mem_array #(.DEPTH(DEPTH), .WIDTH(DATA_WIDTH), .NW(NUM_PORTS), .NR(WIDE), .ZERO_INIT(1'b1)) mem (
      .clk_i(clk_i), .we_i(we), .waddr_i(waddr), .wmask_i('1), .wdata_i(wdata_i), .re_i({WIDE{wide_enable_i}}), .raddr_i(raddr), .rdata_o(rdata));

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      in_range     <= '0;
      wide_valid_o <= 1'b0;
    end else begin
      wide_valid_o <= wide_enable_i;
      if (wide_enable_i)
        for (int k = 0; k < WIDE; k++) in_range[k] <= wide_addr_i[k] < DEPTH;
    end
  end

endmodule
