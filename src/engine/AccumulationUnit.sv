`timescale 1ns / 100ps

// Reduces the P partial tiles into one output tile, then writes it to the mesh SRAM.
// Every cycle it reads the same pixel from all P partial tiles and sums them in a log2(P) adder tree.
// Results leave the tree in pixel order, one per cycle, and are written as they emerge.
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
  localparam int CW = $clog2(PIXELS + 1);
  localparam int ADD_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int LEVELS = $clog2(P);  // adder levels; 0 when there is one partial tile

  // Entries at tree level l: level 0 holds the P partial pixels.
  function automatic int width_at(input int l);
    return (P + (1 << l) - 1) >> l;
  endfunction

  typedef enum logic [1:0] {
    RIDLE,
    RREAD,
    RWAIT,
    RDONE
  } rstate_t;
  rstate_t r_curr;

  logic [CW-1:0] rd_idx;  // pixel being read from all P partial tiles
  logic [CW-1:0] w_idx;  // pixels written so far

  logic read_issue;
  assign read_issue = (r_curr == RREAD);

  always_comb begin
    tile_ren_o  = '0;
    tile_addr_o = '{default: 0};
    if (read_issue)
      for (int k = 0; k < P; k++) begin
        tile_ren_o[k]  = 1'b1;
        tile_addr_o[k] = 32'(rd_idx);
      end
  end

  // A read issued at cycle t presents its data at t+1, so level 0 is valid one cycle behind.
  logic lvl0_v;
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) lvl0_v <= 1'b0;
    else lvl0_v <= read_issue;
  end

  logic [DATA_WIDTH-1:0] lvl_d[LEVELS+1][P];
  logic                  lvl_v[LEVELS+1];
  assign lvl_v[0] = lvl0_v;
  for (genvar k = 0; k < P; k++) begin : L0
    assign lvl_d[0][k] = tile_data_i[k];
  end

  for (genvar l = 0; l < LEVELS; l++) begin : LVL
    localparam int IN_W = width_at(l);
    localparam int OUT_W = width_at(l + 1);
    logic [OUT_W-1:0] done_bits;
    for (genvar m = 0; m < OUT_W; m++) begin : NODE
      if (2 * m + 1 < IN_W) begin : ADD
        fp32Adder adder (
            .clk_i      (clk_i),
            .rstn_i     (rstn_i),
            .valid_i    (lvl_v[l]),
            .A          (lvl_d[l][2*m]),
            .B          (lvl_d[l][2*m+1]),
            .result_o   (lvl_d[l+1][m]),
            .done_o     (done_bits[m]),
            .overflow_o (),
            .underflow_o(),
            .invalid_o  ()
        );
      end else begin : PASS
        // An odd entry out: delay it by the adder latency so it stays aligned with its level.
        logic [DATA_WIDTH-1:0] dly[ADD_LAT];
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
    for (genvar m = OUT_W; m < P; m++) begin : UNUSED
      assign lvl_d[l+1][m] = '0;
    end
    assign lvl_v[l+1] = done_bits[0];
  end

  logic [31:0] local_row, local_col;
  always_comb begin
    local_row = 32'(w_idx) / N;
    local_col = 32'(w_idx) % N;
  end

  assign write_en_o   = lvl_v[LEVELS];
  assign write_data_o = lvl_d[LEVELS][0];
  assign write_addr_o = ((TILE_ROW_OFFSET + local_row) * MATRIX_WIDTH) + (TILE_COL_OFFSET + local_col);
  assign done_o       = (r_curr == RDONE);

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      r_curr <= RIDLE;
      rd_idx <= '0;
      w_idx  <= '0;
    end else begin
      if (write_en_o) w_idx <= w_idx + 1'b1;
      case (r_curr)
        RIDLE: begin
          if (start_i) begin
            rd_idx <= '0;
            w_idx  <= '0;
            r_curr <= RREAD;
          end
        end
        RREAD: begin
          if (rd_idx == CW'(PIXELS - 1)) r_curr <= RWAIT;
          else rd_idx <= rd_idx + 1'b1;
        end
        RWAIT: if (write_en_o && (w_idx == CW'(PIXELS - 1))) r_curr <= RDONE;
        RDONE: if (rearm_i) r_curr <= RIDLE;
        default: r_curr <= RIDLE;
      endcase
    end
  end

endmodule
