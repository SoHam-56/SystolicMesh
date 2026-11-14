`timescale 1ns / 100ps

// Turns a write of WW contiguous words at a word address into mem_array row-port drives: WW at or above ROW_WORDS lands as WW / ROW_WORDS whole
// rows, below it as one row under a word mask. The word address must be a multiple of WW; WW and ROW_WORDS must divide each other.
module row_writer #(
    parameter int ROW_WORDS = 16,
    parameter int WW        = 16,  // words per write
    parameter int DW        = 32,
    parameter int ROWS      = 32,
    parameter int NP        = (WW > ROW_WORDS) ? WW / ROW_WORDS : 1,  // row ports one write drives
    parameter int AW        = (ROWS > 1) ? $clog2(ROWS) : 1,
    parameter int WAW       = $clog2(ROWS * ROW_WORDS)  // word address
) (
    input  logic                            we_i,
    input  logic [WAW-1:0]                  waddr_i,
    input  logic [ WW-1:0][        DW-1:0]  wdata_i,
    output logic [ NP-1:0]                  we_o,
    output logic [ NP-1:0][        AW-1:0]  waddr_o,
    output logic [ NP-1:0][ ROW_WORDS-1:0]  wmask_o,
    output logic [ NP-1:0][ROW_WORDS*DW-1:0] wdata_o
);
  if ((WW >= ROW_WORDS && WW % ROW_WORDS != 0) || (WW < ROW_WORDS && ROW_WORDS % WW != 0)) begin : G_BAD_WW
    $fatal(1, "row_writer: WW %0d and ROW_WORDS %0d must divide each other", WW, ROW_WORDS);
  end
  if (WW >= ROW_WORDS) begin : G_ROWS  // whole rows, one port each
    always_comb
      for (int p = 0; p < NP; p++) begin
        we_o[p]    = we_i;
        waddr_o[p] = AW'(int'(waddr_i) / ROW_WORDS + p);
        wmask_o[p] = '1;
        for (int w = 0; w < ROW_WORDS; w++) wdata_o[p][w*DW+:DW] = wdata_i[p*ROW_WORDS+w];
      end
  end else begin : G_PART  // part of one row: the words at the address's offset in it
    always_comb begin
      automatic int off = int'(waddr_i) % ROW_WORDS;
      we_o[0]    = we_i;
      waddr_o[0] = AW'(int'(waddr_i) / ROW_WORDS);
      wmask_o[0] = '0;
      wdata_o[0] = '0;
      for (int w = 0; w < ROW_WORDS; w++)
        if (w >= off && w < off + WW) begin
          wmask_o[0][w] = 1'b1;
          wdata_o[0][w*DW+:DW] = wdata_i[w-off];
        end
    end
  end
endmodule
