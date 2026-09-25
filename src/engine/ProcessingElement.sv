`timescale 1ns / 100ps

// Output-stationary PE for the pipelined SystolicArray: one product per cycle, sets back to back with no gap.
// Every K products form a set; each set accumulates into its own bank of U partial sums, which the reader combines.
module ProcessingElement #(
    parameter int DATA_WIDTH = 32,
    parameter int K          = 4,  // products per set
    parameter int BANKS      = 3,  // sets held at once: one accumulating, the older ones finishing or being read
    parameter int U          = (K < 6) ? K : 6,  // partial sums per set: the adder latency plus one
    parameter int BW         = (BANKS > 1) ? $clog2(BANKS) : 1
) (
    input  logic                            clk_i,
    input  logic                            rstn_i,
    input  logic [          DATA_WIDTH-1:0] a_i,
    input  logic [          DATA_WIDTH-1:0] b_i,
    input  logic                            v_i,
    output logic [          DATA_WIDTH-1:0] a_o,
    output logic [          DATA_WIDTH-1:0] b_o,
    output logic                            v_o,
    input  logic [                  BW-1:0] rd_bank_i,   // bank the reader looks at
    output logic [U-1:0][DATA_WIDTH-1:0]    partial_o,   // that bank's partial sums
    input  logic                            release_i,   // the reader is done with rd_bank_i
    output logic [               BANKS-1:0] final_o      // per bank: a finished set, all adds written back
);
  localparam int ADD_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int S = ADD_LAT + 1;  // a slot is read again S cycles after its add issues, one after the write-back
  localparam int SW = (U > 1) ? $clog2(U) : 1;
  localparam int CW = $clog2(K + 1);

  initial if (U > S || U > K) $error("ProcessingElement: U=%0d must not exceed min(K=%0d, %0d)", U, K, S);

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      a_o <= '0;
      b_o <= '0;
      v_o <= 1'b0;
    end else begin
      a_o <= a_i;
      b_o <= b_i;
      v_o <= v_i;
    end
  end

  logic [DATA_WIDTH-1:0] prod, sum;
  logic prod_v, sum_v;

  fp32Multiplier MUL (
      .clk_i      (clk_i),
      .rstn_i     (rstn_i),
      .valid_i    (v_i),
      .A          (a_i),
      .B          (b_i),
      .result_o   (prod),
      .done_o     (prod_v),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

  logic [DATA_WIDTH-1:0] acc[BANKS][U];
  logic [BW-1:0] cur;  // bank the next product joins
  logic [SW-1:0] slot;  // partial sum within it
  logic [CW-1:0] n_prod;  // products taken into cur
  logic [BANKS-1:0] taken;  // all K products of the bank issued, not yet released

  // The first product into a slot is added to zero, so a reused bank needs no clear.
  logic [DATA_WIDTH-1:0] add_a;
  assign add_a = (n_prod < CW'(U)) ? '0 : acc[cur][slot];

  fp32Adder ADD (
      .clk_i      (clk_i),
      .rstn_i     (rstn_i),
      .valid_i    (prod_v),
      .A          (add_a),
      .B          (prod),
      .result_o   (sum),
      .done_o     (sum_v),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

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
      taken   <= '0;
      final_o <= '0;
      for (int b = 0; b < BANKS; b++) for (int u = 0; u < U; u++) acc[b][u] <= '0;
    end else begin
      if (sum_v) acc[bank_dly[ADD_LAT-1]][slot_dly[ADD_LAT-1]] <= sum;
      if (prod_v) begin
        if (n_prod == CW'(K - 1)) begin
          taken[cur] <= 1'b1;
          cur        <= (cur == BW'(BANKS - 1)) ? '0 : cur + 1'b1;
          slot       <= '0;
          n_prod     <= '0;
        end else begin
          slot   <= (slot == SW'(U - 1)) ? '0 : slot + 1'b1;
          n_prod <= n_prod + 1'b1;
        end
      end
      for (int b = 0; b < BANKS; b++) if (taken[b] && !pending[b]) final_o[b] <= 1'b1;
      if (release_i) begin
        taken[rd_bank_i]   <= 1'b0;
        final_o[rd_bank_i] <= 1'b0;
      end
    end
  end

  always_comb for (int u = 0; u < U; u++) partial_o[u] = acc[rd_bank_i][u];

`ifndef SYNTHESIS
  a_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (prod_v && n_prod == '0) |-> !taken[cur])
    else $error("ProcessingElement: a set started in bank %0d before the reader released it", cur);
  a_release_final: assert property (@(posedge clk_i) disable iff (!rstn_i) release_i |-> final_o[rd_bank_i])
    else $error("ProcessingElement: bank %0d released before its set was final", rd_bank_i);
`endif

endmodule
