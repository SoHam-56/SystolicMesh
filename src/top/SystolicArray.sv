`timescale 1ns / 100ps

// Pipelined N x N output-stationary tile: C = A * B with A N x K and B K x N, sets back to back.
// Two operand banks let the next set load while one feeds; sets feed with no gap, and each keeps its partials in its own PE bank.
module SystolicArray #(
    parameter int N           = 4,
    parameter int K           = N,  // depth of the product; N for a square tile
    parameter int WEST_WORDS  = K,  // A words per write: one row of A
    parameter int NORTH_WORDS = N,  // B words per write: one row of B
    parameter int BANKS       = 3,  // sets whose partials the PEs hold at once
    parameter int U           = (K < sienna_fmt_pkg::ADD_LATENCY + 1) ? K : sienna_fmt_pkg::ADD_LATENCY + 1,  // partial sums per pixel, combined by the reader
    parameter int COL0        = 0   // mesh column of this tile's column 0, for packed blocks
) (
    input logic clk_i,
    input logic rstn_i,

    input logic                                   north_write_enable_i,
    input logic [NORTH_WORDS-1:0][sienna_fmt_pkg::DATA_WIDTH-1:0] north_write_data_i,
    input logic                                   west_write_enable_i,
    input logic [ WEST_WORDS-1:0][sienna_fmt_pkg::DATA_WIDTH-1:0] west_write_data_i,
    input logic                                   commit_i,      // the operands just written form a pass: queue it
    input logic                                   commit_fresh_i,  // with commit_i: the pass starts a set
    input logic                                   commit_more_i,   // with commit_i: another pass of the same set follows
    input logic [2:0]                             commit_pack_i,   // with commit_i: the pass's pack shift
    output logic                                  load_ready_o,  // an operand bank is free to write

    output logic                               set_final_o,    // the oldest unread set is final in every PE
    output logic                               next_final_o,   // so is the one after it, for a reader about to release
    input  logic                               read_enable_i,
    input  logic [          $clog2(N*N)-1:0]   read_addr_i,
    output logic [U-1:0][sienna_fmt_pkg::ACC_DATA_WIDTH-1:0] read_data_o,    // the pixel's U partial sums, RD_LAT cycles later (one per 64:1 select stage)
    output logic                               read_valid_o,
    input  logic                               release_i,      // the reader is done with the oldest final set
    output logic                               busy_o          // a set is loaded, feeding or unread
);
  localparam int DATA_WIDTH = sienna_fmt_pkg::DATA_WIDTH;  // operands
  localparam int ACC_DATA_WIDTH = sienna_fmt_pkg::ACC_DATA_WIDTH;  // partial sums
  localparam int AD = N * K;  // A and B each hold N*K words per bank
  localparam int BW = (BANKS > 1) ? $clog2(BANKS) : 1;
  localparam int RPW = $clog2(N * N);  // a pixel address
  localparam int RSEL = 6;  // address bits a read stage selects on: at most a 64:1 select per register
  localparam int RD_LAT = (RPW <= RSEL) ? 1 : (RPW + RSEL - 1) / RSEL;  // the read: one cycle per stage (AccumulationUnit's RD_LAT, the same rule)
  localparam int LS = (RD_LAT > 1) ? RD_LAT - 2 : 0;  // the last tree stage before read_data_o
  localparam int RG = (N * N + (1 << RSEL) - 1) >> RSEL;  // registers in the first stage, one per group of 64 pixels
  localparam int KW = (K > 1) ? $clog2(K) : 1;
  localparam int OB_AW = $clog2(2 * AD);  // the operand banks' word address
  // A continuing pass reads the slots its set's previous pass wrote: with K < S a PE reuses each slot every pass, so the pass waits S cycles.
  localparam int S = sienna_fmt_pkg::add_latency(ACC_DATA_WIDTH) + 1;  // a PE slot is ready S cycles after its add issues
  localparam int GW = $clog2(S + 1);

  // Operand banks: A row-major (A[r][kk] at r*K+kk), B row-major (B[kk][c] at kk*N+c), two banks each in one word memory (mem_array).
  // Written WEST_WORDS / NORTH_WORDS words at a time; row r's feed reads A's word (r, kk) and B's word (kk, r) combinationally into a_feed / b_feed.
  logic [$clog2(AD):0] wa, wb;
  logic [WEST_WORDS-1:0] am_we;
  logic [WEST_WORDS-1:0][OB_AW-1:0] am_waddr;
  logic [WEST_WORDS-1:0][DATA_WIDTH-1:0] am_wdata;
  logic [NORTH_WORDS-1:0] bm_we;
  logic [NORTH_WORDS-1:0][OB_AW-1:0] bm_waddr;
  logic [NORTH_WORDS-1:0][DATA_WIDTH-1:0] bm_wdata;
  logic [N-1:0][OB_AW-1:0] am_raddr, bm_raddr;
  logic [N-1:0][DATA_WIDTH-1:0] am_rdata, bm_rdata;
  logic lb;  // bank being written
  logic fb;  // oldest queued bank, next to feed
  logic [1:0] ob_full;  // committed, not yet fed out
  logic [1:0] ob_fresh, ob_more;  // per operand bank: the pass's flags
  logic [2:0] ob_pack[2];  // per operand bank: the pass's pack shift
  // Feed commands: row 0 and column 0 see the command now; row r and column c see it r and c cycles later.
  logic          cmd_v [N];
  logic [KW-1:0] cmd_kk[N];
  logic          cmd_ob[N];
  logic          cmd_f [N];
  logic          cmd_m [N];
  logic [2:0]    cmd_p [N];
  logic feeding;
  logic [KW-1:0] kk;
  logic ob_cur;
  logic [BW-1:0] ab_next;  // accumulator bank the next set lands in, as the PEs count it
  logic [BANKS-1:0] ab_busy;  // a set was fed into the bank and not yet released
  logic launch;  // a queued set starts feeding this cycle
  logic spaced;  // a continuing pass may launch now
  // Skewed feed registers.
  logic [DATA_WIDTH-1:0] a_feed[N], b_feed[N];
  logic v_feed[N], f_feed[N], m_feed[N];
  logic [2:0] p_feed[N];
  // The PE grid.
  logic [DATA_WIDTH-1:0] a_w[N][N+1];  // a_w[r][c] enters PE(r,c) from the west
  logic [DATA_WIDTH-1:0] b_n[N+1][N];  // b_n[r][c] enters PE(r,c) from the north
  logic v_w[N][N+1], f_w[N][N+1], m_w[N][N+1];  // valid and pass flags travel east with A
  logic [2:0] p_w[N][N+1];  // pack shift travels east with A
  logic [U-1:0][ACC_DATA_WIDTH-1:0] part[N][N];
  // RD_LAT > 1: the read as a tree, each stage a <= 64:1 select into registers (one 256:1 select in a cycle was the T16 meshes' 1.054-1.104 ns,
  // acc_unit rd_idx -> tile read_data_o): stage k holds one word per 64^(k+1) pixels, beside the address and valid a stage on.
  logic [U-1:0][ACC_DATA_WIDTH-1:0] rd_st[LS+1][RG];
  logic [RPW-1:0]                   rd_sa[LS+1];
  logic                             rd_sv[LS+1];
  logic [BANKS-1:0] pe_final[N*N];
  // The reader's bank, and per bank, registered, whether its set is final in every PE.
  logic [BW-1:0] rb;  // oldest unread accumulator bank
  logic [BW-1:0] rb_n;
  logic [BANKS-1:0] bank_final_c, bank_final;

`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial begin
    if ((AD % WEST_WORDS) != 0 || (AD % NORTH_WORDS) != 0)
      $error("SystolicArray: write widths %0d/%0d must divide %0d", WEST_WORDS, NORTH_WORDS, AD);
  end
  // the PEs' slot count: fewer slots reuse one within a pass before its add lands, and the spacing below assumes U = K under S
  initial if (U != ((K < S) ? K : S)) $error("SystolicArray: U=%0d must be min(K=%0d, %0d)", U, K, S);
`endif

  assign load_ready_o = !ob_full[lb];

  if (K < S) begin : G_SPACE
    logic [GW-1:0] since;  // cycles since the last launch, saturating at S
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) since <= GW'(S);
      else if (launch) since <= GW'(1);
      else if (int'(since) < S) since <= since + 1'b1;
    assign spaced = ob_fresh[fb] || (int'(since) >= S);
  end else begin : G_NO_SPACE  // a pass is at least S cycles long, so back-to-back passes are already S apart
    assign spaced = 1'b1;
  end

  // A pass that starts a set needs a free partial-sum bank; a continuing pass adds into the one its set holds, once it is spaced.
  assign launch = ob_full[fb] && (!ob_fresh[fb] || !ab_busy[ab_next]) && (!feeding || kk == KW'(K - 1)) && spaced;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      wa      <= '0;
      wb      <= '0;
      lb      <= 1'b0;
      fb      <= 1'b0;
      ob_full <= '0;
      ob_fresh <= '0;
      ob_more <= '0;
      ob_pack[0] <= '0;
      ob_pack[1] <= '0;
      feeding <= 1'b0;
      kk      <= '0;
      ob_cur  <= 1'b0;
      ab_next <= '0;
      for (int r = 0; r < N; r++) begin
        cmd_v[r]  <= 1'b0;
        cmd_kk[r] <= '0;
        cmd_ob[r] <= 1'b0;
        cmd_f[r]  <= 1'b0;
        cmd_m[r]  <= 1'b0;
        cmd_p[r]  <= '0;
      end
    end else begin
      if (west_write_enable_i && wa < AD) wa <= wa + WEST_WORDS;  // the words land in a_mem below
      if (north_write_enable_i && wb < AD) wb <= wb + NORTH_WORDS;
      if (commit_i) begin
        ob_full[lb] <= 1'b1;
        ob_fresh[lb] <= commit_fresh_i;
        ob_more[lb] <= commit_more_i;
        ob_pack[lb] <= commit_pack_i;
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
        if (!ob_more[fb]) ab_next <= (ab_next == BW'(BANKS - 1)) ? '0 : ab_next + 1'b1;  // the set's last pass
      end else if (feeding) begin
        if (kk == KW'(K - 1)) feeding <= 1'b0;
        else kk <= kk + 1'b1;
      end
      cmd_v[0]  <= launch || (feeding && kk != KW'(K - 1));
      cmd_kk[0] <= launch ? '0 : kk + 1'b1;
      cmd_ob[0] <= launch ? fb : ob_cur;
      cmd_f[0]  <= launch ? ob_fresh[fb] : ob_fresh[ob_cur];
      cmd_m[0]  <= launch ? ob_more[fb] : ob_more[ob_cur];
      cmd_p[0]  <= launch ? ob_pack[fb] : ob_pack[ob_cur];
      for (int r = 1; r < N; r++) begin
        cmd_v[r]  <= cmd_v[r-1];
        cmd_kk[r] <= cmd_kk[r-1];
        cmd_ob[r] <= cmd_ob[r-1];
        cmd_f[r]  <= cmd_f[r-1];
        cmd_m[r]  <= cmd_m[r-1];
        cmd_p[r]  <= cmd_p[r-1];
      end
      // The bank is free once the last row and column have taken their last operand.
      if (cmd_v[N-1] && cmd_kk[N-1] == KW'(K - 1)) ob_full[cmd_ob[N-1]] <= 1'b0;
    end
  end

  // ── Skewed feed registers ─────────────────────────────────────────────
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int r = 0; r < N; r++) begin
        v_feed[r] <= 1'b0;
        f_feed[r] <= 1'b0;
        m_feed[r] <= 1'b0;
        p_feed[r] <= '0;
      end
    end else begin
      for (int r = 0; r < N; r++) begin
        v_feed[r] <= cmd_v[r];
        f_feed[r] <= cmd_f[r];
        m_feed[r] <= cmd_m[r];
        p_feed[r] <= cmd_p[r];
      end
    end
  end
  // The feed words: a_mem and b_mem read synchronously on cmd_v (an SRAM's read), so their registered words come with v_feed, the
  // cycle the combinational read's register gave them (step C of the memory plan: no latency change).
  always_comb
    for (int r = 0; r < N; r++) begin
      a_feed[r] = v_feed[r] ? am_rdata[r] : '0;
      b_feed[r] = v_feed[r] ? bm_rdata[r] : '0;  // column r of B
    end
  always_comb begin
    for (int c = 0; c < WEST_WORDS; c++) begin
      am_we[c]    = west_write_enable_i && wa < AD;
      am_waddr[c] = OB_AW'(int'(lb) * AD + int'(wa) + c);
      am_wdata[c] = west_write_data_i[c];
    end
    for (int c = 0; c < NORTH_WORDS; c++) begin
      bm_we[c]    = north_write_enable_i && wb < AD;
      bm_waddr[c] = OB_AW'(int'(lb) * AD + int'(wb) + c);
      bm_wdata[c] = north_write_data_i[c];
    end
    for (int r = 0; r < N; r++) begin
      am_raddr[r] = OB_AW'(int'(cmd_ob[r]) * AD + r * K + int'(cmd_kk[r]));
      bm_raddr[r] = OB_AW'(int'(cmd_ob[r]) * AD + int'(cmd_kk[r]) * N + r);
    end
  end
  logic [N-1:0] feed_re;  // row r reads on its command
  always_comb for (int r = 0; r < N; r++) feed_re[r] = cmd_v[r];
  mem_array #(.DEPTH(2 * AD), .WIDTH(DATA_WIDTH), .NW(WEST_WORDS), .NR(N), .SYNC_READ(1'b1)) a_mem (
      .clk_i(clk_i), .we_i(am_we), .waddr_i(am_waddr), .wmask_i('1), .wdata_i(am_wdata), .re_i(feed_re), .raddr_i(am_raddr), .rdata_o(am_rdata));
  mem_array #(.DEPTH(2 * AD), .WIDTH(DATA_WIDTH), .NW(NORTH_WORDS), .NR(N), .SYNC_READ(1'b1)) b_mem (
      .clk_i(clk_i), .we_i(bm_we), .waddr_i(bm_waddr), .wmask_i('1), .wdata_i(bm_wdata), .re_i(feed_re), .raddr_i(bm_raddr), .rdata_o(bm_rdata));

  // ── PE grid ───────────────────────────────────────────────────────────
  for (genvar r = 0; r < N; r++) begin : FEED
    assign a_w[r][0] = a_feed[r];
    assign v_w[r][0] = v_feed[r];
    assign f_w[r][0] = f_feed[r];
    assign m_w[r][0] = m_feed[r];
    assign p_w[r][0] = p_feed[r];
    assign b_n[0][r] = b_feed[r];
  end

  for (genvar r = 0; r < N; r++) begin : ROW
    for (genvar c = 0; c < N; c++) begin : COL
      ProcessingElement #(
          .K         (K),
          .BANKS     (BANKS),
          .U         (U),
          .BW        (BW),
          .COL       (COL0 + c)
      ) pe (
          .clk_i    (clk_i),
          .rstn_i   (rstn_i),
          .a_i      (a_w[r][c]),
          .b_i      (b_n[r][c]),
          .v_i      (v_w[r][c]),
          .fresh_i  (f_w[r][c]),
          .more_i   (m_w[r][c]),
          .a_o      (a_w[r][c+1]),
          .b_o      (b_n[r+1][c]),
          .v_o      (v_w[r][c+1]),
          .fresh_o  (f_w[r][c+1]),
          .more_o   (m_w[r][c+1]),
          .pack_i   (p_w[r][c]),
          .pack_o   (p_w[r][c+1]),
          .partial_o(part[r][c]),
          .release_i(release_i),
          .final_o  (pe_final[r*N+c])
      );
    end
  end

  // Per bank, registered: the set in it is final in every PE (one cycle after the last PE's final_o; cleared with the release). The AND
  // over N*N PEs then ends in a flop instead of running on into the mesh's reducer start, and the bank select picks a flop, not a tree.
  assign rb_n = (rb == BW'(BANKS - 1)) ? '0 : rb + 1'b1;
  always_comb
    for (int b = 0; b < BANKS; b++) begin
      bank_final_c[b] = ab_busy[b];
      for (int p = 0; p < N * N; p++) bank_final_c[b] &= pe_final[p][b];
    end
  assign set_final_o  = bank_final[rb];
  assign next_final_o = bank_final[rb_n];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ab_busy      <= '0;
      bank_final   <= '0;
      rb           <= '0;
      read_data_o  <= '0;
      read_valid_o <= 1'b0;
    end else begin
      if (launch && ob_fresh[fb]) ab_busy[ab_next] <= 1'b1;
      bank_final <= bank_final_c;
      if (release_i) begin
        ab_busy[rb] <= 1'b0;
        bank_final[rb] <= 1'b0;
        rb <= (rb == BW'(BANKS - 1)) ? '0 : rb + 1'b1;
      end
      if (RD_LAT == 1) begin
        read_valid_o <= read_enable_i;
        if (read_enable_i) read_data_o <= part[read_addr_i/N][read_addr_i%N];
      end else begin
        read_valid_o <= rd_sv[LS];
        if (rd_sv[LS]) read_data_o <= rd_st[LS][int'(rd_sa[LS]) >> (RSEL * (LS + 1))];
      end
    end
  end
  // RD_LAT > 1: stage 0 selects within each group of 64 pixels on the address' low 6 bits, each later stage within groups of 64 of the
  // stage before on the next 6, and read_data_o picks among the last stage's words (at 256 pixels: four groups, then the quarter).
  if (RD_LAT > 1) begin : G_TREE
    // The read address as this tile's own register: the reducer reads 0 .. N*N - 1 in order, back to back or after a pause from 0, so a
    // counter here equals read_addr_i on every read (a_rd_loc) and the stage-0 selects fan out from inside the tile (the reducer's rd_idx
    // into every group register of the tiles it reads was the T16 meshes' 0.99-1.01 ns at 0.8).
    // One copy per group of 64 pixels (rd_loc_g), each driving only its group's select: one counter for the whole tile still fanned out
    // to every group register (tile rd_loc -> rd_st, int8 N16 T16 0.947 at 0.8). The copies are equal by construction; the flow keeps them
    // apart (no register merging on rd_loc_g), else they fold back into one.
    logic [RPW-1:0] rd_loc;
    logic [RPW-1:0] rd_loc_g[RG];
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) begin
        rd_loc <= '0;
        for (int j = 0; j < RG; j++) rd_loc_g[j] <= '0;
      end else begin
        rd_loc <= read_enable_i ? rd_loc + 1'b1 : '0;
        for (int j = 0; j < RG; j++) rd_loc_g[j] <= read_enable_i ? rd_loc_g[j] + 1'b1 : '0;
      end
`ifndef SYNTHESIS
    a_rd_loc: assert property (@(posedge clk_i) disable iff (!rstn_i) read_enable_i |-> rd_loc == read_addr_i)
      else $error("SystolicArray: a_rd_loc: the local read address %0d is not read_addr_i %0d", rd_loc, read_addr_i);
    for (genvar j = 0; j < RG; j++) begin : G_LOC_CHK
      a_rd_loc_g: assert property (@(posedge clk_i) disable iff (!rstn_i) rd_loc_g[j] == rd_loc)
        else $error("SystolicArray: a_rd_loc_g: group %0d's read address copy %0d is not %0d", j, rd_loc_g[j], rd_loc);
    end
`else
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_addr;
    assign unused_addr = ^read_addr_i;  // the tile counts the address itself; only the check reads the port
    /* verilator lint_on UNUSEDSIGNAL */
`endif
    always_ff @(posedge clk_i or negedge rstn_i)
      if (!rstn_i) for (int k = 0; k <= LS; k++) rd_sv[k] <= 1'b0;
      else begin
        rd_sv[0] <= read_enable_i;
        for (int k = 1; k <= LS; k++) rd_sv[k] <= rd_sv[k-1];
      end
    always_ff @(posedge clk_i) begin  // no reset: used only with its stage's valid
      if (read_enable_i) begin
        rd_sa[0] <= rd_loc;
        for (int j = 0; j < RG; j++) rd_st[0][j] <= part[(j * 64 + (int'(rd_loc_g[j]) % 64)) / N][(j * 64 + (int'(rd_loc_g[j]) % 64)) % N];
      end
      for (int k = 1; k <= LS; k++)
        if (rd_sv[k-1]) begin
          rd_sa[k] <= rd_sa[k-1];
          for (int j = 0; j < RG; j++)
            if (j * 64 + ((int'(rd_sa[k-1]) >> (RSEL * k)) % 64) < RG) rd_st[k][j] <= rd_st[k-1][j*64+((int'(rd_sa[k-1])>>(RSEL*k))%64)];
        end
    end
  end else begin : G_READ1
    /* verilator lint_off UNUSEDSIGNAL */
    logic unused_rd;
    assign unused_rd = ^{rd_st[0][0], rd_sa[0], rd_sv[0]};
    /* verilator lint_on UNUSEDSIGNAL */
    assign rd_sv[0] = 1'b0;
    assign rd_sa[0] = '0;
    for (genvar j = 0; j < RG; j++) begin : G_Z
      assign rd_st[0][j] = '0;
    end
  end

  assign busy_o = (|ob_full) || feeding || (|ab_busy);

`ifndef SYNTHESIS
  int gap_q;  // cycles since the last launch, for a_continue_spaced
  int early_passes;  // cover: continuing passes that were ready to launch under S cycles after the previous launch (the spacing holds them)
  logic early_q;  // the queued continuing pass was ready early
  // The last row may be written in the commit cycle itself.
  a_commit_loaded: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                    commit_i |-> (int'(wa) + (west_write_enable_i ? WEST_WORDS : 0) == AD &&
                                                  int'(wb) + (north_write_enable_i ? NORTH_WORDS : 0) == AD))
    else $error("SystolicArray: commit before both operands were fully written");
  a_write_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (west_write_enable_i || north_write_enable_i) |-> !ob_full[lb])
    else $error("SystolicArray: write into an operand bank that is still queued");
  a_commit_free: assert property (@(posedge clk_i) disable iff (!rstn_i) commit_i |-> !ob_full[lb])
    else $error("SystolicArray: commit into an operand bank that is still queued");
  a_continue_held: assert property (@(posedge clk_i) disable iff (!rstn_i) (launch && !ob_fresh[fb]) |-> ab_busy[ab_next])
    else $error("SystolicArray: a continuing pass launched with no set holding its bank");
  a_read_final: assert property (@(posedge clk_i) disable iff (!rstn_i) (read_enable_i || release_i) |-> set_final_o)
    else $error("SystolicArray: read or release with no final set");
  // The PEs' a_slot_ready one level up, on its own counter: a continuing pass launches at least S cycles after the previous launch.
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) gap_q <= S;
    else if (launch) gap_q <= 1;
    else if (gap_q < S) gap_q <= gap_q + 1;
  a_continue_spaced: assert property (@(posedge clk_i) disable iff (!rstn_i) (launch && !ob_fresh[fb]) |-> gap_q >= S)
    else $error("SystolicArray: a continuing pass launched %0d cycles after the previous one, its slots need %0d", gap_q, S);
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) begin
      early_passes <= 0;
      early_q <= 1'b0;
    end else if (launch) begin
      if (early_q || (!ob_fresh[fb] && gap_q < S)) early_passes <= early_passes + 1;
      early_q <= 1'b0;
    end else if (ob_full[fb] && !ob_fresh[fb] && (!feeding || kk == KW'(K - 1)) && gap_q < S) early_q <= 1'b1;
`endif

endmodule
