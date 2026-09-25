`timescale 1ns / 100ps

// Pipelined N x N output-stationary tile: C = A * B with A N x K and B K x N, sets back to back.
// Two operand banks let the next set load while one feeds; sets feed with no gap, and each keeps its partials in its own PE bank.
module SystolicArray #(
    parameter int N           = 4,
    parameter int K           = N,  // depth of the product; N for a square tile
    parameter int DATA_WIDTH  = 32,
    parameter int WEST_WORDS  = K,  // A words per write: one row of A
    parameter int NORTH_WORDS = N,  // B words per write: one row of B
    parameter int BANKS       = 3,  // sets whose partials the PEs hold at once
    parameter int U           = (K < 6) ? K : 6  // partial sums per pixel, combined by the reader
) (
    input logic clk_i,
    input logic rstn_i,

    input logic                                   north_write_enable_i,
    input logic [NORTH_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                                   west_write_enable_i,
    input logic [ WEST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                                   commit_i,      // the operands just written form a set: queue it
    output logic                                  load_ready_o,  // an operand bank is free to write

    output logic                               set_final_o,    // the oldest unread set is final in every PE
    output logic                               next_final_o,   // so is the one after it, for a reader about to release
    input  logic                               read_enable_i,
    input  logic [          $clog2(N*N)-1:0]   read_addr_i,
    output logic [U-1:0][DATA_WIDTH-1:0]       read_data_o,    // the pixel's U partial sums, one cycle later
    output logic                               read_valid_o,
    input  logic                               release_i,      // the reader is done with the oldest final set
    output logic                               busy_o          // a set is loaded, feeding or unread
);
  localparam int AD = N * K;  // A and B each hold N*K words per bank
  localparam int BW = (BANKS > 1) ? $clog2(BANKS) : 1;
  localparam int KW = (K > 1) ? $clog2(K) : 1;

  initial begin
    if ((AD % WEST_WORDS) != 0 || (AD % NORTH_WORDS) != 0)
      $error("SystolicArray: write widths %0d/%0d must divide %0d", WEST_WORDS, NORTH_WORDS, AD);
  end

  // ── Operand banks: A row-major (A[r][kk] at r*K+kk), B row-major (B[kk][c] at kk*N+c) ──
  logic [DATA_WIDTH-1:0] a_mem[2][AD];
  logic [DATA_WIDTH-1:0] b_mem[2][AD];
  logic [$clog2(AD):0] wa, wb;
  logic lb;  // bank being written
  logic fb;  // oldest queued bank, next to feed
  logic [1:0] ob_full;  // committed, not yet fed out

  assign load_ready_o = !ob_full[lb];

  // Feed commands: row 0 and column 0 see the command now; row r and column c see it r and c cycles later.
  logic          cmd_v [N];
  logic [KW-1:0] cmd_kk[N];
  logic          cmd_ob[N];
  logic feeding;
  logic [KW-1:0] kk;
  logic ob_cur;
  logic [BW-1:0] ab_next;  // accumulator bank the next set lands in, as the PEs count it
  logic [BANKS-1:0] ab_busy;  // a set was fed into the bank and not yet released
  logic launch;  // a queued set starts feeding this cycle

  assign launch = ob_full[fb] && !ab_busy[ab_next] && (!feeding || kk == KW'(K - 1));

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wa      <= '0;
      wb      <= '0;
      lb      <= 1'b0;
      fb      <= 1'b0;
      ob_full <= '0;
      feeding <= 1'b0;
      kk      <= '0;
      ob_cur  <= 1'b0;
      ab_next <= '0;
      for (int r = 0; r < N; r++) begin
        cmd_v[r]  <= 1'b0;
        cmd_kk[r] <= '0;
        cmd_ob[r] <= 1'b0;
      end
    end else begin
      if (west_write_enable_i && wa < AD) begin
        for (int c = 0; c < WEST_WORDS; c++) a_mem[lb][int'(wa)+c] <= west_write_data_i[c];
        wa <= wa + WEST_WORDS;
      end
      if (north_write_enable_i && wb < AD) begin
        for (int c = 0; c < NORTH_WORDS; c++) b_mem[lb][int'(wb)+c] <= north_write_data_i[c];
        wb <= wb + NORTH_WORDS;
      end
      if (commit_i) begin
        ob_full[lb] <= 1'b1;
        lb <= ~lb;
        wa <= '0;
        wb <= '0;
      end

      // Row 0 walks kk through the set; a queued set follows the last kk with no gap.
      if (launch) begin
        feeding <= 1'b1;
        kk      <= '0;
        ob_cur  <= fb;
        fb      <= ~fb;
        ab_next <= (ab_next == BW'(BANKS - 1)) ? '0 : ab_next + 1'b1;
      end else if (feeding) begin
        if (kk == KW'(K - 1)) feeding <= 1'b0;
        else kk <= kk + 1'b1;
      end
      cmd_v[0]  <= launch || (feeding && kk != KW'(K - 1));
      cmd_kk[0] <= launch ? '0 : kk + 1'b1;
      cmd_ob[0] <= launch ? fb : ob_cur;
      for (int r = 1; r < N; r++) begin
        cmd_v[r]  <= cmd_v[r-1];
        cmd_kk[r] <= cmd_kk[r-1];
        cmd_ob[r] <= cmd_ob[r-1];
      end
      // The bank is free once the last row and column have taken their last operand.
      if (cmd_v[N-1] && cmd_kk[N-1] == KW'(K - 1)) ob_full[cmd_ob[N-1]] <= 1'b0;
    end
  end

  // ── Skewed feed registers ─────────────────────────────────────────────
  logic [DATA_WIDTH-1:0] a_feed[N], b_feed[N];
  logic v_feed[N];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int r = 0; r < N; r++) begin
        a_feed[r] <= '0;
        b_feed[r] <= '0;
        v_feed[r] <= 1'b0;
      end
    end else begin
      for (int r = 0; r < N; r++) begin
        v_feed[r] <= cmd_v[r];
        a_feed[r] <= cmd_v[r] ? a_mem[cmd_ob[r]][r*K+int'(cmd_kk[r])] : '0;
        b_feed[r] <= cmd_v[r] ? b_mem[cmd_ob[r]][int'(cmd_kk[r])*N+r] : '0;  // column r of B
      end
    end
  end

  // ── PE grid ───────────────────────────────────────────────────────────
  logic [DATA_WIDTH-1:0] a_w[N][N+1];  // a_w[r][c] enters PE(r,c) from the west
  logic [DATA_WIDTH-1:0] b_n[N+1][N];  // b_n[r][c] enters PE(r,c) from the north
  logic v_w[N][N+1];
  logic [U-1:0][DATA_WIDTH-1:0] part[N][N];
  logic [BANKS-1:0] pe_final[N*N];
  logic [BW-1:0] rb;  // oldest unread accumulator bank

  for (genvar r = 0; r < N; r++) begin : FEED
    assign a_w[r][0] = a_feed[r];
    assign v_w[r][0] = v_feed[r];
    assign b_n[0][r] = b_feed[r];
  end

  for (genvar r = 0; r < N; r++) begin : ROW
    for (genvar c = 0; c < N; c++) begin : COL
      ProcessingElement #(
          .DATA_WIDTH(DATA_WIDTH),
          .K         (K),
          .BANKS     (BANKS),
          .U         (U),
          .BW        (BW)
      ) pe (
          .clk_i    (clk_i),
          .rstn_i   (rstn_i),
          .a_i      (a_w[r][c]),
          .b_i      (b_n[r][c]),
          .v_i      (v_w[r][c]),
          .a_o      (a_w[r][c+1]),
          .b_o      (b_n[r+1][c]),
          .v_o      (v_w[r][c+1]),
          .rd_bank_i(rb),
          .partial_o(part[r][c]),
          .release_i(release_i),
          .final_o  (pe_final[r*N+c])
      );
    end
  end

  logic all_final, all_next;
  logic [BW-1:0] rb_n;
  assign rb_n = (rb == BW'(BANKS - 1)) ? '0 : rb + 1'b1;
  always_comb begin
    all_final = ab_busy[rb];
    all_next  = ab_busy[rb_n];
    for (int p = 0; p < N * N; p++) begin
      all_final &= pe_final[p][rb];
      all_next  &= pe_final[p][rb_n];
    end
  end
  assign set_final_o  = all_final;
  assign next_final_o = all_next;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ab_busy      <= '0;
      rb           <= '0;
      read_data_o  <= '0;
      read_valid_o <= 1'b0;
    end else begin
      if (launch) ab_busy[ab_next] <= 1'b1;
      if (release_i) begin
        ab_busy[rb] <= 1'b0;
        rb <= (rb == BW'(BANKS - 1)) ? '0 : rb + 1'b1;
      end
      read_valid_o <= read_enable_i;
      if (read_enable_i) read_data_o <= part[read_addr_i/N][read_addr_i%N];
    end
  end

  assign busy_o = (|ob_full) || feeding || (|ab_busy);

`ifndef SYNTHESIS
  // The last row may be written in the commit cycle itself.
  a_commit_loaded: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                    commit_i |-> (int'(wa) + (west_write_enable_i ? WEST_WORDS : 0) == AD &&
                                                  int'(wb) + (north_write_enable_i ? NORTH_WORDS : 0) == AD))
    else $error("SystolicArray: commit before both operands were fully written");
  a_write_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (west_write_enable_i || north_write_enable_i) |-> !ob_full[lb])
    else $error("SystolicArray: write into an operand bank that is still queued");
  a_commit_free: assert property (@(posedge clk_i) disable iff (!rstn_i) commit_i |-> !ob_full[lb])
    else $error("SystolicArray: commit into an operand bank that is still queued");
  a_read_final: assert property (@(posedge clk_i) disable iff (!rstn_i) (read_enable_i || release_i) |-> set_final_o)
    else $error("SystolicArray: read or release with no final set");
`endif

endmodule
