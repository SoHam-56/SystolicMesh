`timescale 1ns / 100ps

// Reduces the P partial tiles into one output tile, then writes it to the mesh SRAM.
//
// The P accumulation steps for a single pixel are loop-carried, but the PIXELS pixels are
// independent, so the pixels are rotated through the one pipelined adder (C-slow) instead of
// stalling on its latency once per step. One add is issued per cycle; a pixel is only revisited
// a full round later, which is why the round period has a floor of ADD_LAT+1.
module AccumulationUnit #(
    parameter P = 8,
    parameter N = 4,
    parameter DATA_WIDTH = 32,
    parameter MATRIX_WIDTH = 32,
    parameter TILE_ROW_OFFSET = 0,
    parameter TILE_COL_OFFSET = 0
) (
    input logic clk_i,
    rstn_i,
    start_i,
    rearm_i,

    input  logic [P-1:0][DATA_WIDTH-1:0] tile_data_i,
    input  logic [P-1:0]                 tile_valid_i,
    output logic [P-1:0]                 tile_ren_o,
    output logic [P-1:0][          31:0] tile_addr_o,

    output logic                  write_en_o,
    output logic [          31:0] write_addr_o,
    output logic [DATA_WIDTH-1:0] write_data_o,

    output logic done_o
);
  localparam int PIXELS = N * N;
  localparam int PXW = $clog2(PIXELS);
  localparam int ADD_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int MIN_PER = ADD_LAT + 1;  // a pixel must not be revisited before its write-back
  localparam int PERIOD = (PIXELS > MIN_PER) ? PIXELS : MIN_PER;
  localparam int CW = $clog2(PERIOD + 1);

  logic [DATA_WIDTH-1:0] acc[PIXELS];

  logic [CW-1:0] cyc;  // position within the current round
  logic [  31:0] k_idx;  // which partial tile this round consumes
  logic [PXW-1:0] w_idx;  // pixel being written out
  logic [   3:0] drain_cnt;

  typedef enum logic [2:0] {
    RIDLE,
    ROUND,
    DRAIN,
    WRITE,
    RDONE
  } rstate_t;
  rstate_t r_curr;

  logic [PXW-1:0] cyc_px;
  logic read_issue;
  assign cyc_px = cyc[PXW-1:0];
  assign read_issue = (r_curr == ROUND) && (cyc < PIXELS);

  // A read issued at cycle t presents its data at t+1, so the add is issued one cycle behind.
  logic iss_v;
  logic [PXW-1:0] iss_p;
  logic [   31:0] iss_k;  // k must lag with the data, or the last add of a round reads tile k+1

  logic [ADD_LAT-1:0] d_v;
  logic [PXW-1:0] d_p[ADD_LAT];

  logic add_done;
  logic [DATA_WIDTH-1:0] add_res, op_a, op_b;

  assign op_a = acc[iss_p];
  assign op_b = tile_data_i[iss_k];

  fp32Adder adder (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .valid_i(iss_v),
      .A(op_a),
      .B(op_b),
      .result_o(add_res),
      .done_o(add_done),
      .overflow_o(),
      .underflow_o(),
      .invalid_o()
  );

  logic [31:0] local_row, local_col, global_addr;

  always_comb begin
    local_row   = w_idx / N;
    local_col   = w_idx % N;
    global_addr = ((TILE_ROW_OFFSET + local_row) * MATRIX_WIDTH) + (TILE_COL_OFFSET + local_col);
  end

  always_comb begin
    tile_ren_o  = '0;
    tile_addr_o = '{default: 0};
    if (read_issue) begin
      tile_ren_o[k_idx]  = 1'b1;
      tile_addr_o[k_idx] = cyc_px;
    end
  end

  assign write_en_o   = (r_curr == WRITE);
  assign write_addr_o = global_addr;
  assign write_data_o = acc[w_idx];
  assign done_o       = (r_curr == RDONE);

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      r_curr    <= RIDLE;
      cyc       <= '0;
      k_idx     <= '0;
      w_idx     <= '0;
      drain_cnt <= '0;
      iss_v     <= 1'b0;
      iss_p     <= '0;
      iss_k     <= '0;
      d_v       <= '0;
      for (int i = 0; i < PIXELS; i++) acc[i] <= '0;
      for (int i = 0; i < ADD_LAT; i++) d_p[i] <= '0;
    end else begin
      // Read issue -> add issue -> ADD_LAT stages -> write-back, each one cycle apart.
      iss_v  <= read_issue;
      iss_p  <= cyc_px;
      iss_k  <= k_idx;
      d_v    <= {d_v[ADD_LAT-2:0], iss_v};
      d_p[0] <= iss_p;
      for (int i = 1; i < ADD_LAT; i++) d_p[i] <= d_p[i-1];

      if (d_v[ADD_LAT-1]) acc[d_p[ADD_LAT-1]] <= add_res;

      case (r_curr)
        RIDLE: begin
          if (start_i) begin
            for (int i = 0; i < PIXELS; i++) acc[i] <= '0;
            cyc    <= '0;
            k_idx  <= '0;
            w_idx  <= '0;
            r_curr <= ROUND;
          end
        end

        ROUND: begin
          if (cyc == PERIOD - 1) begin
            cyc <= '0;
            if (k_idx == P - 1) begin
              drain_cnt <= '0;
              r_curr    <= DRAIN;
            end else begin
              k_idx <= k_idx + 1'b1;
            end
          end else begin
            cyc <= cyc + 1'b1;
          end
        end

        DRAIN: begin
          // Let every add still inside the adder retire before the results are read out.
          if (drain_cnt == ADD_LAT + 2) begin
            w_idx  <= '0;
            r_curr <= WRITE;
          end else begin
            drain_cnt <= drain_cnt + 1'b1;
          end
        end

        WRITE: begin
          if (w_idx == PIXELS - 1) r_curr <= RDONE;
          else w_idx <= w_idx + 1'b1;
        end

        // Was latched with no exit at all - the comment claiming the mesh resets
        // this unit between matmuls was never true, there was no reset port.
        RDONE: if (rearm_i) r_curr <= RIDLE;

        default: r_curr <= RIDLE;
      endcase
    end
  end

endmodule
