`timescale 1ns / 100ps

// Synchronous N x N output-stationary tile: C = A * B with A N x K and B K x N, both held locally.
// Row r of A and column c of B enter r and c cycles late, and operands move one PE per cycle, so no handshake is needed.
module SystolicArray #(
    parameter int N           = 4,
    parameter int K           = N,  // depth of the product; N for a square tile
    parameter int DATA_WIDTH  = 32,
    parameter int WEST_WORDS  = K,  // A words per write: one row of A
    parameter int NORTH_WORDS = N   // B words per write: one row of B
) (
    input logic clk_i,
    input logic rstn_i,
    input logic start_matrix_mult_i,
    input logic rearm_i,

    input logic                                  north_write_enable_i,
    input logic [NORTH_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                                  north_write_reset_i,
    input logic                                  west_write_enable_i,
    input logic [ WEST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                                  west_write_reset_i,

    output logic north_queue_empty_o,
    output logic west_queue_empty_o,
    output logic matrix_mult_complete_o,

    input  logic                   read_enable_i,
    input  logic [$clog2(N*N)-1:0] read_addr_i,
    output logic [ DATA_WIDTH-1:0] read_data_o,
    output logic                   read_valid_o,

    output logic collection_complete_o,
    output logic collection_active_o
);
  localparam int AD = N * K;  // A and B each hold N*K words
  localparam int STEPS = K + N - 1;  // feed cycles: K products plus the skew of the last row or column
  localparam int SCW = $clog2(STEPS + 1);

  initial begin
    if ((AD % WEST_WORDS) != 0 || (AD % NORTH_WORDS) != 0)
      $error("SystolicArray: write widths %0d/%0d must divide %0d", WEST_WORDS, NORTH_WORDS, AD);
  end

  // A row-major (A[r][kk] at r*K+kk), B row-major (B[kk][c] at kk*N+c).
  logic [DATA_WIDTH-1:0] a_mem[AD];
  logic [DATA_WIDTH-1:0] b_mem[AD];
  logic [$clog2(AD):0] wa, wb;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wa <= '0;
      wb <= '0;
    end else begin
      if (west_write_reset_i) wa <= '0;
      else if (west_write_enable_i && wa < AD) begin
        for (int c = 0; c < WEST_WORDS; c++) a_mem[int'(wa)+c] <= west_write_data_i[c];
        wa <= wa + WEST_WORDS;
      end
      if (north_write_reset_i) wb <= '0;
      else if (north_write_enable_i && wb < AD) begin
        for (int c = 0; c < NORTH_WORDS; c++) b_mem[int'(wb)+c] <= north_write_data_i[c];
        wb <= wb + NORTH_WORDS;
      end
    end
  end
  assign west_queue_empty_o  = (wa == 0);
  assign north_queue_empty_o = (wb == 0);

  // Skewed feed: at step s, row r takes A[r][s-r] and column c takes B[s-c][c] when that index is in range.
  logic feeding;
  logic [SCW-1:0] s;
  logic [DATA_WIDTH-1:0] a_feed[N], b_feed[N];
  logic v_feed[N];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      feeding <= 1'b0;
      s <= '0;
      for (int r = 0; r < N; r++) begin
        a_feed[r] <= '0;
        b_feed[r] <= '0;
        v_feed[r] <= 1'b0;
      end
    end else begin
      if (start_matrix_mult_i) begin
        feeding <= 1'b1;
        s <= '0;
      end else if (feeding) begin
        if (s == SCW'(STEPS - 1)) feeding <= 1'b0;
        s <= s + 1'b1;
      end
      for (int r = 0; r < N; r++) begin
        automatic int kk = int'(s) - r;
        automatic bit live = feeding && !start_matrix_mult_i && kk >= 0 && kk < K;
        v_feed[r] <= live;
        a_feed[r] <= live ? a_mem[r*K+kk] : '0;
        b_feed[r] <= live ? b_mem[kk*N+r] : '0;  // column r of B
      end
    end
  end

  logic [DATA_WIDTH-1:0] a_w[N][N+1];  // a_w[r][c] enters PE(r,c) from the west
  logic [DATA_WIDTH-1:0] b_n[N+1][N];  // b_n[r][c] enters PE(r,c) from the north
  logic v_w[N][N+1];
  logic [DATA_WIDTH-1:0] res[N][N];
  logic [N*N-1:0] pe_done;

  for (genvar r = 0; r < N; r++) begin : FEED
    assign a_w[r][0] = a_feed[r];
    assign v_w[r][0] = v_feed[r];
    assign b_n[0][r] = b_feed[r];
  end

  for (genvar r = 0; r < N; r++) begin : ROW
    for (genvar c = 0; c < N; c++) begin : COL
      logic unused_v;
      ProcessingElement #(
          .DATA_WIDTH(DATA_WIDTH),
          .K         (K)
      ) pe (
          .clk_i   (clk_i),
          .rstn_i  (rstn_i),
          .start_i (start_matrix_mult_i),
          .a_i     (a_w[r][c]),
          .b_i     (b_n[r][c]),
          .v_i     (v_w[r][c]),
          .a_o     (a_w[r][c+1]),
          .b_o     (b_n[r+1][c]),
          .v_o     (v_w[r][c+1]),
          .result_o(res[r][c]),
          .done_o  (pe_done[r*N+c])
      );
    end
  end

  // Complete once every PE has its sum, until the mesh re-arms for the next set.
  logic ran;
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) ran <= 1'b0;
    else if (start_matrix_mult_i) ran <= 1'b1;
    else if (rearm_i) ran <= 1'b0;
  end
  assign collection_complete_o  = ran && (&pe_done);
  assign collection_active_o    = ran && !(&pe_done);
  assign matrix_mult_complete_o = collection_complete_o;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      read_data_o  <= '0;
      read_valid_o <= 1'b0;
    end else begin
      read_valid_o <= read_enable_i;
      if (read_enable_i) read_data_o <= res[read_addr_i/N][read_addr_i%N];
    end
  end

`ifndef SYNTHESIS
  a_start_idle: assert property (@(posedge clk_i) disable iff (!rstn_i) start_matrix_mult_i |-> !feeding)
    else $error("SystolicArray: started while still feeding the previous matmul");
  a_loaded: assert property (@(posedge clk_i) disable iff (!rstn_i) start_matrix_mult_i |-> (wa == AD && wb == AD))
    else $error("SystolicArray: started before both operands were fully written");
`endif

endmodule
