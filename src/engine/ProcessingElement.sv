`timescale 1ns / 100ps

// Output-stationary PE for SystolicArray: takes one product every cycle and passes its operands on a cycle later.
// Products rotate through S partial sums so an add never waits on the previous one; the partials are summed at the end.
module ProcessingElement #(
    parameter int DATA_WIDTH = 32,
    parameter int K          = 4    // products per matmul
) (
    input  logic                  clk_i,
    input  logic                  rstn_i,
    input  logic                  start_i,   // clears the sums for a new matmul
    input  logic [DATA_WIDTH-1:0] a_i,
    input  logic [DATA_WIDTH-1:0] b_i,
    input  logic                  v_i,
    output logic [DATA_WIDTH-1:0] a_o,
    output logic [DATA_WIDTH-1:0] b_o,
    output logic                  v_o,
    output logic [DATA_WIDTH-1:0] result_o,
    output logic                  done_o
);
  localparam int ADD_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int S = ADD_LAT + 1;  // a slot's sum is written back S cycles after its add issues
  localparam int U = (K < S) ? K : S;  // partial sums actually used
  localparam int SW = $clog2(S);
  localparam int CW = $clog2(K + 1);

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

  typedef enum logic [1:0] {P_ACC, P_ISSUE, P_WAIT, P_DONE} pstate_t;
  pstate_t st;

  logic [DATA_WIDTH-1:0] acc[S];
  logic [SW-1:0] slot;  // partial sum the next product joins
  logic [CW-1:0] n_prod;  // products taken this matmul
  logic [3:0] inflight;  // adds issued and not yet written back
  logic [SW:0] width, m;  // combine: live partials, next pair to add

  // One adder, shared by the accumulate and the final combine.
  logic add_v;
  logic [DATA_WIDTH-1:0] add_a, add_b;
  logic [SW-1:0] add_tag, tag_dly[ADD_LAT];
  always_comb begin
    add_v = 1'b0;
    add_a = acc[slot];
    add_b = prod;
    add_tag = slot;
    if (st == P_ACC) add_v = prod_v;
    else if (st == P_ISSUE && m < (width >> 1)) begin
      add_v = 1'b1;
      add_a = acc[SW'(2*m)];
      add_b = acc[SW'(2*m+1)];
      add_tag = SW'(m);
    end
  end

  fp32Adder ADD (
      .clk_i      (clk_i),
      .rstn_i     (rstn_i),
      .valid_i    (add_v),
      .A          (add_a),
      .B          (add_b),
      .result_o   (sum),
      .done_o     (sum_v),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) for (int i = 0; i < ADD_LAT; i++) tag_dly[i] <= '0;
    else begin
      tag_dly[0] <= add_tag;
      for (int i = 1; i < ADD_LAT; i++) tag_dly[i] <= tag_dly[i-1];
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      st       <= P_ACC;
      slot     <= '0;
      n_prod   <= '0;
      inflight <= '0;
      width    <= '0;
      m        <= '0;
      for (int i = 0; i < S; i++) acc[i] <= '0;
    end else if (start_i) begin
      st       <= P_ACC;
      slot     <= '0;
      n_prod   <= '0;
      inflight <= '0;
      m        <= '0;
      for (int i = 0; i < S; i++) acc[i] <= '0;
    end else begin
      inflight <= inflight + {3'b0, add_v} - {3'b0, sum_v};
      if (sum_v) acc[tag_dly[ADD_LAT-1]] <= sum;
      case (st)
        P_ACC: begin
          if (prod_v) begin
            slot   <= (slot == SW'(S - 1)) ? '0 : slot + 1'b1;
            n_prod <= n_prod + 1'b1;
          end
          // Every product added and every add written back: the partials are final.
          if (n_prod == CW'(K) && inflight == '0) begin
            width <= (SW + 1)'(U);
            m     <= '0;
            st    <= (U == 1) ? P_DONE : P_ISSUE;
          end
        end
        // One pairwise add per cycle into the low slots; an odd partial moves down unchanged.
        P_ISSUE: begin
          if (m + 1'b1 >= (width >> 1)) begin
            if (width[0]) acc[SW'(width>>1)] <= acc[SW'(width-1)];
            width <= (width + 1'b1) >> 1;
            m     <= '0;
            st    <= P_WAIT;
          end else m <= m + 1'b1;
        end
        P_WAIT: if (inflight == '0 && !sum_v) st <= (width == 1) ? P_DONE : P_ISSUE;
        P_DONE: ;
        default: st <= P_ACC;
      endcase
    end
  end

  assign result_o = acc[0];
  assign done_o   = (st == P_DONE);

`ifndef SYNTHESIS
  a_no_late_product: assert property (@(posedge clk_i) disable iff (!rstn_i) prod_v |-> (st == P_ACC))
    else $error("ProcessingElement: a product arrived after the partial sums were being combined");
  a_slot_free: assert property (@(posedge clk_i) disable iff (!rstn_i) (st == P_ACC && prod_v) |-> (inflight < 4'(S)))
    else $error("ProcessingElement: more adds in flight than partial sums");
`endif

endmodule
