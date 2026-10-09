`timescale 1ns / 100ps

// Output-stationary PE for the pipelined SystolicArray: one product per cycle, sets back to back with no gap.
// Every K products form a pass; a set is one or more passes, accumulated into its own bank of U partial sums, which the reader combines.
// A packed set (pack_i != 0) counts only the products in this PE's column block of K >> pack_i, so each block sums as its job alone.
module ProcessingElement #(
    parameter int EXP_W      = 8,   // the build's format: fp32 8/23, bf16 8/7, int8 0/7
    parameter int MAN_W      = 23,
    parameter int DATA_WIDTH = 1 + EXP_W + MAN_W,  // operands
    parameter int ACC_W      = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // products and sums: int32 in int8, fp32 in every float format
    parameter int K          = 4,  // products per set
    parameter int BANKS      = 3,  // sets held at once: one accumulating, the older ones finishing or being read
    parameter int U          = (K < sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1) ? K : sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1,  // partial sums per set: the adder latency plus one
    parameter int BW         = (BANKS > 1) ? $clog2(BANKS) : 1,
    parameter int COL        = 0  // this PE's column in the whole mesh: its packed block is COL / (K >> pack)
) (
    input  logic                         clk_i,
    input  logic                         rstn_i,
    input  logic [       DATA_WIDTH-1:0] a_i,
    input  logic [       DATA_WIDTH-1:0] b_i,
    input  logic                         v_i,
    input  logic                         fresh_i,     // with v_i: this pass starts a set, its first U counted products add to 0
    input  logic                         more_i,      // with v_i: another pass of the same set follows this one
    input  logic [                  2:0] pack_i,      // with v_i: the set's pack shift; 0 is an unpacked set
    output logic [       DATA_WIDTH-1:0] a_o,
    output logic [       DATA_WIDTH-1:0] b_o,
    output logic                         v_o,
    output logic                         fresh_o,
    output logic                         more_o,
    output logic [                  2:0] pack_o,
    input  logic [               BW-1:0] rd_bank_i,   // bank the reader looks at
    output logic [U-1:0][     ACC_W-1:0] partial_o,   // that bank's partial sums; a slot no product reached reads +0
    input  logic                         release_i,   // the reader is done with rd_bank_i
    output logic [            BANKS-1:0] final_o      // per bank: a finished set, all adds written back
);
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+ADD_LAT
  localparam int MUL_LAT = sienna_fmt_pkg::mul_lat(EXP_W, MAN_W);  // valid_i at t, done_o at t+MUL_LAT
  localparam int S = ADD_LAT + 1;  // a slot is read again S cycles after its add issues, one after the write-back
  localparam int SW = (U > 1) ? $clog2(U) : 1;
  localparam int CW = $clog2(K + 1);
  localparam int LGK = (K > 1) ? $clog2(K) : 1;

`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if (U > S || U > K) $error("ProcessingElement: U=%0d must not exceed min(K=%0d, %0d)", U, K, S);
  initial if (COL >= K) $error("ProcessingElement: COL=%0d must be below K=%0d", COL, K);
`endif

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      a_o <= '0;
      b_o <= '0;
      v_o <= 1'b0;
      fresh_o <= 1'b0;
      more_o <= 1'b0;
      pack_o <= '0;
    end else begin
      a_o <= a_i;
      b_o <= b_i;
      v_o <= v_i;
      fresh_o <= fresh_i;
      more_o <= more_i;
      pack_o <= pack_i;
    end
  end

  // The input's place in its pass, and whether it lies in this PE's block (always, unpacked).
  logic [CW-1:0] k_in;
  logic in_blk;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) k_in <= '0;
    else if (v_i) k_in <= (k_in == CW'(K - 1)) ? '0 : k_in + 1'b1;
  assign in_blk = (pack_i == '0) || (((int'(k_in) ^ COL) >> (LGK - int'(pack_i))) == 0);

  // The pass flags, delayed to meet their product out of the multiplier; v_d ticks for every product, counted or not.
  logic fresh_d[MUL_LAT], more_d[MUL_LAT], v_d[MUL_LAT], inb_d[MUL_LAT];
  logic [2:0] pack_d[MUL_LAT];
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < MUL_LAT; i++) begin
        fresh_d[i] <= 1'b0;
        more_d[i] <= 1'b0;
        v_d[i] <= 1'b0;
        inb_d[i] <= 1'b0;
        pack_d[i] <= '0;
      end
    end else begin
      fresh_d[0] <= fresh_i;
      more_d[0] <= more_i;
      v_d[0] <= v_i;
      inb_d[0] <= in_blk;
      pack_d[0] <= pack_i;
      for (int i = 1; i < MUL_LAT; i++) begin
        fresh_d[i] <= fresh_d[i-1];
        more_d[i] <= more_d[i-1];
        v_d[i] <= v_d[i-1];
        inb_d[i] <= inb_d[i-1];
        pack_d[i] <= pack_d[i-1];
      end
    end
  end
  logic prod_fresh, prod_more, prod_tick;
  logic [2:0] prod_pack;
  assign prod_fresh = fresh_d[MUL_LAT-1];
  assign prod_more  = more_d[MUL_LAT-1];
  assign prod_tick  = v_d[MUL_LAT-1];
  assign prod_pack  = pack_d[MUL_LAT-1];

  logic [ACC_W-1:0] prod, sum;
  logic prod_v, sum_v;  // prod_v: a counted product, only those reach the multiplier

  logic [ACC_W-1:0] acc[BANKS][U];
  logic [U-1:0] wr_mask[BANKS];  // per bank: slots a counted product reached this set
  logic [BW-1:0] cur;  // bank the next product joins
  logic [SW-1:0] slot;  // partial sum within it
  logic [CW-1:0] n_prod;  // products of the pass seen, counted or not
  logic [CW-1:0] u_cnt;  // counted products of the pass
  logic [BANKS-1:0] taken;  // all K products of the bank issued, not yet released

  // The first counted product into a slot of a new set is added to zero, so a reused bank needs no clear; later passes add on.
  logic [ACC_W-1:0] add_a;
  assign add_a = (prod_fresh && u_cnt < CW'(U)) ? '0 : acc[cur][slot];

  // The multiplier by operand format, the adder by accumulator format: bf16 multiplies exactly into fp32 and sums in fp32; int8 into int32, wrapping.
  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "ProcessingElement: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC_W
    $fatal(1, "ProcessingElement: ACC_W=%0d is not sienna_fmt_pkg::acc_w(%0d, %0d)", ACC_W, EXP_W, MAN_W);
  end else if (sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_FP32
    fp32Multiplier MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i), .result_o(prod), .done_o(prod_v),
                        .overflow_o(), .underflow_o(), .invalid_o());
    fp32Adder ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum), .done_o(sum_v),
                   .overflow_o(), .underflow_o(), .invalid_o());
  end else if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT
    logic [2*DATA_WIDTH-1:0] prod_w;  // the full signed product
    intMultiplier #(.W(DATA_WIDTH)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i), .result_o(prod_w),
        .done_o(prod_v));
    assign prod = {{(ACC_W - 2 * DATA_WIDTH){prod_w[2*DATA_WIDTH-1]}}, prod_w};  // sign-extended to the accumulator
    intAdder #(.W(ACC_W)) ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum),
        .done_o(sum_v));
  end else begin : G_FP
    if ($bits(prod) != 32) begin : G_BAD_PROD  // fpMulWiden's fp32 product would be truncated silently
      $fatal(1, "ProcessingElement: prod is %0d bits, fpMulWiden's fp32 product is 32", $bits(prod));
    end
    fpMulWiden #(.EXP_W(EXP_W), .MAN_W(MAN_W)) MUL (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(v_i && in_blk), .A(a_i), .B(b_i),
        .result_o(prod), .done_o(prod_v), .overflow_o(), .underflow_o(), .invalid_o());  // prod is X before its first done_o; read with prod_v
    fp32Adder ADD (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(prod_v), .A(add_a), .B(prod), .result_o(sum), .done_o(sum_v),
                   .overflow_o(), .underflow_o(), .invalid_o());
  end

  // Where each add in flight writes back, and whether it is still in flight.
  logic [BW-1:0] bank_dly[ADD_LAT];
  logic [SW-1:0] slot_dly[ADD_LAT];
  logic          v_dly   [ADD_LAT];

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < ADD_LAT; i++) begin
        bank_dly[i] <= '0;
        slot_dly[i] <= '0;
        v_dly[i]    <= 1'b0;
      end
    end else begin
      bank_dly[0] <= cur;
      slot_dly[0] <= slot;
      v_dly[0]    <= prod_v;
      for (int i = 1; i < ADD_LAT; i++) begin
        bank_dly[i] <= bank_dly[i-1];
        slot_dly[i] <= slot_dly[i-1];
        v_dly[i]    <= v_dly[i-1];
      end
    end
  end

  logic [BANKS-1:0] pending;  // an add into the bank is issuing or in flight
  always_comb begin
    pending = '0;
    if (prod_v) pending[cur] = 1'b1;
    for (int i = 0; i < ADD_LAT; i++) if (v_dly[i]) pending[bank_dly[i]] = 1'b1;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      cur     <= '0;
      slot    <= '0;
      n_prod  <= '0;
      u_cnt   <= '0;
      taken   <= '0;
      final_o <= '0;
      for (int b = 0; b < BANKS; b++) begin
        wr_mask[b] <= '0;
        for (int u = 0; u < U; u++) acc[b][u] <= '0;
      end
    end else begin
      if (sum_v) acc[bank_dly[ADD_LAT-1]][slot_dly[ADD_LAT-1]] <= sum;
      // A set's first product clears its bank's mask; each slot a counted product of its first pass reaches is marked.
      if (prod_tick && n_prod == '0 && prod_fresh) wr_mask[cur] <= prod_v ? (U'(1) << slot) : '0;
      else if (prod_v && prod_fresh && u_cnt < CW'(U)) wr_mask[cur][slot] <= 1'b1;
      if (prod_v) begin
        slot  <= (slot == SW'(U - 1)) ? '0 : slot + 1'b1;
        u_cnt <= u_cnt + 1'b1;
      end
      if (prod_tick) begin
        if (n_prod == CW'(K - 1)) begin
          n_prod <= '0;
          u_cnt  <= '0;
          if (!prod_more) begin  // the set's last pass; a continuing set keeps its bank and the slot keeps turning
            taken[cur] <= 1'b1;
            cur        <= (cur == BW'(BANKS - 1)) ? '0 : cur + 1'b1;
            slot       <= '0;
          end
        end else n_prod <= n_prod + 1'b1;
      end
      for (int b = 0; b < BANKS; b++) if (taken[b] && !pending[b]) final_o[b] <= 1'b1;
      if (release_i) begin
        taken[rd_bank_i]   <= 1'b0;
        final_o[rd_bank_i] <= 1'b0;
      end
    end
  end

  always_comb for (int u = 0; u < U; u++) partial_o[u] = wr_mask[rd_bank_i][u] ? acc[rd_bank_i][u] : '0;

`ifndef SYNTHESIS
  a_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_tick && n_prod == '0) |-> !taken[cur])
    else $error("ProcessingElement: a set started in bank %0d before the reader released it", cur);
  a_flags_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) prod_v == (v_d[MUL_LAT-1] && inb_d[MUL_LAT-1]))
    else $error("ProcessingElement: the pass flags are out of step with the multiplier");
  a_fresh_slot0: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_v && u_cnt == '0 && prod_fresh) |-> slot == '0)
    else $error("ProcessingElement: a new set started part way through a bank's slots");
  a_release_final: assert property (@(posedge clk_i) disable iff (!rstn_i) release_i |-> final_o[rd_bank_i])
    else $error("ProcessingElement: bank %0d released before its set was final", rd_bank_i);
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) v_i |-> int'(pack_i) < LGK)
    else $error("ProcessingElement: pack shift %0d leaves blocks narrower than 2 of K=%0d", pack_i, K);
  a_pack_count: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                 (prod_tick && n_prod == CW'(K - 1)) |-> (int'(u_cnt) + int'(prod_v)) == (K >> int'(prod_pack)))
    else $error("ProcessingElement: a pass counted %0d products, its block holds %0d", int'(u_cnt) + int'(prod_v),
                K >> int'(prod_pack));
`endif

endmodule
