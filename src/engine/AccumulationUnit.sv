`timescale 1ns / 100ps

// Reduces the P partial sums of every pixel of one output tile and writes the tile to the mesh SRAM.
// Every cycle it reads one pixel's P partials (U per array, from every depth slice) and sums them in a log2(P) adder tree.
// A new set may start as soon as the last pixel of the previous one has been read; each write carries its own result bank.
module AccumulationUnit #(
    parameter P = 8,
    parameter N = 4,
    parameter EXP_W = 8,  // the build's format: fp32 8/23, bf16 8/7, int8 0/7
    parameter MAN_W = 23,
    parameter ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // partials, bias and the tree: int32 in int8, fp32 in every float format
    parameter OUT_W = sienna_fmt_pkg::out_w(EXP_W, MAN_W),  // written results: int32 in int8, the format's own width in floats
    parameter MATRIX_WIDTH = 32,
    parameter TILE_ROW_OFFSET = 0,
    parameter TILE_COL_OFFSET = 0,
    parameter RESULT_BANKS = 2,
    parameter RBW = (RESULT_BANKS > 1) ? $clog2(RESULT_BANKS) : 1
) (
    input logic clk_i,
    input logic rstn_i,
    input logic start_i,  // read the arrays' oldest final set now
    input logic [RBW-1:0] out_bank_i,  // result bank this set is written to
    input logic [P-1:0][ACC_W-1:0] tile_data_i,
    input logic [N-1:0][ACC_W-1:0] bias_i,  // at start: the set's bias for this tile's N columns, zero for none
    output logic [ACC_W-1:0] bias_word_o,  // bias of the pixel read last cycle, one of the P inputs
    output logic rd_en_o,
    output logic [$clog2(N*N)-1:0] rd_addr_o,
    output logic read_done_o,  // one cycle: the last pixel was read, the arrays may release the set
    output logic ready_o,  // not reading; a start is taken
    output logic write_en_o,
    output logic [31:0] write_addr_o,  // includes the result bank offset
    output logic [OUT_W-1:0] write_data_o,
    output logic written_o,  // one cycle: the last pixel of a set was written
    output logic busy_o  // reading, or pixels still in the tree
);
  localparam int PIXELS = N * N;
  localparam int PW = (PIXELS > 1) ? $clog2(PIXELS) : 1;
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);  // fp32Adder (every float) or intAdder: valid_i at t, done_o at t+ADD_LAT

  if (!sienna_fmt_pkg::supported(EXP_W, MAN_W)) begin : G_BAD_FORMAT
    $fatal(1, "AccumulationUnit: unsupported format EXP_W=%0d MAN_W=%0d", EXP_W, MAN_W);
  end else if (ACC_W != sienna_fmt_pkg::acc_w(EXP_W, MAN_W)) begin : G_BAD_ACC_W
    $fatal(1, "AccumulationUnit: ACC_W=%0d is not sienna_fmt_pkg::acc_w(%0d, %0d)", ACC_W, EXP_W, MAN_W);
  end else if (OUT_W != sienna_fmt_pkg::out_w(EXP_W, MAN_W)) begin : G_BAD_OUT_W  // a result word would be truncated or padded silently
    $fatal(1, "AccumulationUnit: OUT_W=%0d is not sienna_fmt_pkg::out_w(%0d, %0d)", OUT_W, EXP_W, MAN_W);
  end
  localparam int LEVELS = $clog2(P);  // adder levels; 0 when there is one partial
  localparam int LAT = 1 + LEVELS * ADD_LAT;  // read issue to tree output
  localparam int BANK_OFFSET = MATRIX_WIDTH * MATRIX_WIDTH;

  // Entries at tree level l: level 0 holds the P partials.
  function automatic int width_at(input int l);
    return (P + (1 << l) - 1) >> l;
  endfunction

  logic reading;
  logic [PW-1:0] rd_idx;
  logic [RBW-1:0] rd_bank;
  logic [N-1:0][ACC_W-1:0] rd_bias;

  assign ready_o   = !reading || read_done_o;  // the next set may start as the last pixel is read
  assign rd_en_o   = reading;
  assign rd_addr_o = rd_idx;
  assign read_done_o = reading && (rd_idx == PW'(PIXELS - 1));

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      reading <= 1'b0;
      rd_idx  <= '0;
      rd_bank <= '0;
    end else if (start_i && ready_o) begin
      reading <= 1'b1;
      rd_idx  <= '0;
      rd_bank <= out_bank_i;
      rd_bias <= bias_i;
    end else if (reading) begin
      if (rd_idx == PW'(PIXELS - 1)) reading <= 1'b0;
      else rd_idx <= rd_idx + 1'b1;
    end
  end

  // The bias word joins the tree with the pixel's partials, which arrive one cycle after the read.
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) bias_word_o <= '0;
    else bias_word_o <= rd_bias[int'(rd_idx) % N];
  end

  // Pixel index and result bank travel beside the data, LAT cycles from read to write.
  logic          tag_v   [LAT];
  logic [PW-1:0] tag_idx [LAT];
  logic [RBW-1:0] tag_bank[LAT];
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int s = 0; s < LAT; s++) begin
        tag_v[s]    <= 1'b0;
        tag_idx[s]  <= '0;
        tag_bank[s] <= '0;
      end
    end else begin
      tag_v[0]    <= reading;
      tag_idx[0]  <= rd_idx;
      tag_bank[0] <= rd_bank;
      for (int s = 1; s < LAT; s++) begin
        tag_v[s]    <= tag_v[s-1];
        tag_idx[s]  <= tag_idx[s-1];
        tag_bank[s] <= tag_bank[s-1];
      end
    end
  end

  // A read issued at cycle t presents its data at t+1, so level 0 is valid one cycle behind.
  logic [ACC_W-1:0] lvl_d[LEVELS+1][P];
  logic                  lvl_v[LEVELS+1];
  assign lvl_v[0] = tag_v[0];
  for (genvar k = 0; k < P; k++) begin : L0
    assign lvl_d[0][k] = tile_data_i[k];
  end

  for (genvar l = 0; l < LEVELS; l++) begin : LVL
    localparam int IN_W = width_at(l);
    localparam int NXT_W = width_at(l + 1);
    logic [NXT_W-1:0] done_bits;
    for (genvar m = 0; m < NXT_W; m++) begin : NODE
      if (2 * m + 1 < IN_W) begin : ADD
        if (sienna_fmt_pkg::is_int(EXP_W)) begin : G_INT
          intAdder #(.W(ACC_W)) adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]), .A(lvl_d[l][2*m]), .B(lvl_d[l][2*m+1]),
                                       .result_o(lvl_d[l+1][m]), .done_o(done_bits[m]));
        end else begin : G_FP32  // the accumulator's format: fp32 in every float build
          fp32Adder adder (.clk_i(clk_i), .rstn_i(rstn_i), .valid_i(lvl_v[l]), .A(lvl_d[l][2*m]), .B(lvl_d[l][2*m+1]),
                           .result_o(lvl_d[l+1][m]), .done_o(done_bits[m]), .overflow_o(), .underflow_o(), .invalid_o());
        end
      end else begin : PASS
        // An odd entry out: delay it by the adder latency so it stays aligned with its level.
        logic [ACC_W-1:0] dly[ADD_LAT];
        logic                  vdly[ADD_LAT];
        always_ff @(posedge clk_i or negedge rstn_i) begin
          if (!rstn_i) begin
            for (int s = 0; s < ADD_LAT; s++) begin
              dly[s]  <= '0;
              vdly[s] <= 1'b0;
            end
          end else begin
            dly[0]  <= lvl_d[l][2*m];
            vdly[0] <= lvl_v[l];
            for (int s = 1; s < ADD_LAT; s++) begin
              dly[s]  <= dly[s-1];
              vdly[s] <= vdly[s-1];
            end
          end
        end
        assign lvl_d[l+1][m] = dly[ADD_LAT-1];
        assign done_bits[m]  = vdly[ADD_LAT-1];
      end
    end
    for (genvar m = NXT_W; m < P; m++) begin : UNUSED
      assign lvl_d[l+1][m] = '0;
    end
    assign lvl_v[l+1] = done_bits[0];
  end

  logic [PW-1:0] w_idx;
  logic [RBW-1:0] w_bank;
  assign w_idx  = tag_idx[LAT-1];
  assign w_bank = tag_bank[LAT-1];

  assign write_en_o   = lvl_v[LEVELS];
  if (!sienna_fmt_pkg::is_int(EXP_W) && !sienna_fmt_pkg::is_fp32(EXP_W, MAN_W)) begin : G_NARROW  // bf16: each fp32 sum rounded once, to nearest even
    fpNarrow #(.EXP_W(EXP_W), .MAN_W(MAN_W)) NARROW (.x_i(lvl_d[LEVELS][0]), .y_o(write_data_o));
  end else begin : G_SAME  // fp32 and int8 write the sum itself
    assign write_data_o = lvl_d[LEVELS][0];
  end
  assign write_addr_o = int'(w_bank) * BANK_OFFSET +
                        ((TILE_ROW_OFFSET + int'(w_idx) / N) * MATRIX_WIDTH) + (TILE_COL_OFFSET + int'(w_idx) % N);
  assign written_o    = write_en_o && (w_idx == PW'(PIXELS - 1));

  logic in_tree;
  always_comb begin
    in_tree = 1'b0;
    for (int s = 0; s < LAT; s++) in_tree |= tag_v[s];
  end
  assign busy_o = reading || in_tree;

`ifndef SYNTHESIS
  a_tag_aligned: assert property (@(posedge clk_i) disable iff (!rstn_i) lvl_v[LEVELS] == tag_v[LAT-1])
    else $error("AccumulationUnit: tree output and its pixel tag are out of step");
`endif

endmodule
