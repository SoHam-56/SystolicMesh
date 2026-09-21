`timescale 1ns / 100ps

module PEMesh #(
    parameter N          = 2,  // Array size (NxN)
    parameter DATA_WIDTH = 32  // Data width for each PE
) (
    input logic clk_i,
    input logic rstn_i,

    input logic [DATA_WIDTH-1:0] north_i[0:N-1],  // Data inputs (top row - North boundary)
    input logic [DATA_WIDTH-1:0] west_i [0:N-1],  // Weight inputs (left column - West boundary)

    input logic inputs_valid_i,  // Single input valid for top-left PE
    input logic last_element_i,                              // Pulse that indicates last element has been released from InputQueue

    output logic [DATA_WIDTH-1:0] south_o[0:N-1],  // Data outputs (bottom row - South boundary)
    output logic [DATA_WIDTH-1:0] east_o [0:N-1],  // Weight outputs (right column - East boundary)

    output logic passthrough_valid_o[0:N-1][0:N-1],
    output logic done_o,
    output logic [N-1:0] drain_o
);

  wire [DATA_WIDTH-1:0] north_connections[0:N][0:N-1];  // Data connections (North-South flow)
  wire [DATA_WIDTH-1:0] west_connections[0:N-1][0:N];  // Weight connections (West-East flow)
  wire inputs_valid_internal [0:N-1][0:N-1];              // Valid signal connections (follow the systolic flow pattern)
  wire fwd_valid [0:N-1][0:N-1];                          // Early passthrough valid, released before the MAC retires
  wire pe_accept [0:N-1][0:N-1];                          // High when a PE takes its inputs, clears that PE's join
  wire pe_idle [0:N-1][0:N-1];                            // High when a PE has nothing in flight
  wire accumulator_valid_connections[0:N-1][0:N];  // Accumulator valid connections (West-East flow)
  wire last_element_horizontal [0:N-1][0:N];              // last_element connections (horizontal flow in bottom row)

  logic [N-1:0] col_shift;
  logic wave_active;
  logic select_accumulator[0:N-1][0:N-1];

  logic last_element_seen;
  logic waiting_for_passthrough;

  logic matrix_mult_done_ff;
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) matrix_mult_done_ff <= 1'b0;
    else matrix_mult_done_ff <= done_o;
  end
  wire all_pe_idle;
  logic all_idle_r;
  always_comb begin
    all_idle_r = 1'b1;
    foreach (pe_idle[i, j]) if (!pe_idle[i][j]) all_idle_r = 1'b0;
  end
  assign all_pe_idle = all_idle_r;

  // done_o's rising edge is one cycle wide, so the request is latched and the wave released
  // once every PE is actually idle. Gating the edge directly would drop it.
  logic drain_pending;
  wire  start_wave = drain_pending & all_pe_idle & ~wave_active;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) drain_pending <= 1'b0;
    else if (done_o & ~matrix_mult_done_ff) drain_pending <= 1'b1;
    else if (start_wave) drain_pending <= 1'b0;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      col_shift   <= '0;
      wave_active <= 1'b0;
    end else begin
      if (start_wave) begin
        col_shift   <= {1'b1, {(N - 1) {1'b0}}};  // load a '1' into MSB
        wave_active <= 1'b1;
      end else if (wave_active) begin
        col_shift <= col_shift >> 1;
        if (col_shift == '0) wave_active <= 1'b0;  // stop after shift register empties
      end
    end
  end

  // Broadcast col_shift to select_accumulator
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) foreach (select_accumulator[i, j]) select_accumulator[i][j] <= 1'b0;
    else foreach (select_accumulator[i, j]) select_accumulator[i][j] <= col_shift[j];
  end

  genvar row, col;
  generate
    for (row = 0; row < N; row = row + 1) begin : gen_row
      for (col = 0; col < N; col = col + 1) begin : gen_col
        ProcessingElement #(
            .DATA_WIDTH(DATA_WIDTH)
        ) pe_inst (
            .clk_i(clk_i),
            .rstn_i(rstn_i),
            .north_i(north_connections[row][col]),
            .west_i(west_connections[row][col]),
            .inputs_valid_i(inputs_valid_internal[row][col]),
            .last_element_i(last_element_horizontal[row][col]),
            .select_accumulator_i(select_accumulator[row][col]),
            .accumulator_valid_i(accumulator_valid_connections[row][col]),
            .south_o(north_connections[row+1][col]),
            .east_o(west_connections[row][col+1]),
            .passthrough_valid_o(passthrough_valid_o[row][col]),
            .fwd_valid_o(fwd_valid[row][col]),
            .accept_o(pe_accept[row][col]),
            .idle_o(pe_idle[row][col]),
            .accumulator_valid_o(accumulator_valid_connections[row][col+1]),
            .last_element_east_o(last_element_horizontal[row][col+1])
        );
      end
    end
  endgenerate

  // Connect boundary inputs and outputs
  generate
    for (col = 0; col < N; col = col + 1) begin : gen_weight_boundary
      assign north_connections[0][col] = north_i[col];                          // Top boundary: Connect external weight inputs to first row
      assign south_o[col] = north_connections[N][col];                          // Bottom boundary: Connect last row outputs to external outputs
    end

    for (row = 0; row < N; row = row + 1) begin : gen_data_boundary
      assign west_connections[row][0] = west_i[row];                            // Left boundary: Connect external data inputs to first column
      assign east_o[row] = west_connections[row][N];                            // Right boundary: Connect last column outputs to external outputs
    end

    // Accumulator valid boundary connections (West-East flow)
    for (row = 0; row < N; row = row + 1) begin : gen_accumulator_valid_boundary
      assign accumulator_valid_connections[row][0] = 1'b0;                      // Left boundary: No accumulator valid input from outside
      assign drain_o[row] = accumulator_valid_connections[row][N];              // Right boundary: accumulator_valid_o from rightmost PE goes nowhere
    end
  endgenerate

  // Connect last_element signals
  generate
    for (row = 0; row < N; row = row + 1) begin : gen_last_element_boundary
      if (row == N - 1)
        assign last_element_horizontal[row][0] = last_element_i;  // Bottom row: Connect external last_element_i only to leftmost PE
      else assign last_element_horizontal[row][0] = 1'b0;  // Other rows: No last_element input
    end
  endgenerate

  // Connect inputs_valid signals following systolic flow pattern
  generate
    for (row = 0; row < N; row = row + 1) begin : gen_valid_row
      for (col = 0; col < N; col = col + 1) begin : gen_valid_col
        // Chained off fwd_valid, not passthrough_valid, so a neighbour need not wait out this
        // PE's multiply and add. Each incoming valid is latched and held until the PE accepts:
        // the pulses are one cycle wide and, once PEs run at independent offsets, they neither
        // coincide with each other nor with the consumer being idle.
        if (row == 0 && col == 0)
          assign inputs_valid_internal[row][col] = inputs_valid_i;                    // Top-left PE gets external inputs_valid
        else if (row == 0) begin : gen_join_w
          reg west_seen;
          assign inputs_valid_internal[row][col] = west_seen | fwd_valid[row][col-1];
          always @(posedge clk_i or negedge rstn_i) begin
            if (!rstn_i) west_seen <= 1'b0;
            else if (pe_accept[row][col]) west_seen <= 1'b0;
            else if (fwd_valid[row][col-1]) west_seen <= 1'b1;
          end
        end
        else if (col == 0) begin : gen_join_n
          reg north_seen;
          assign inputs_valid_internal[row][col] = north_seen | fwd_valid[row-1][col];
          always @(posedge clk_i or negedge rstn_i) begin
            if (!rstn_i) north_seen <= 1'b0;
            else if (pe_accept[row][col]) north_seen <= 1'b0;
            else if (fwd_valid[row-1][col]) north_seen <= 1'b1;
          end
        end
        else begin : gen_join
          reg north_seen, west_seen;
          wire north_now = north_seen | fwd_valid[row-1][col];
          wire west_now = west_seen | fwd_valid[row][col-1];
          assign inputs_valid_internal[row][col] = north_now & west_now;
          always @(posedge clk_i or negedge rstn_i) begin
            if (!rstn_i) begin
              north_seen <= 1'b0;
              west_seen  <= 1'b0;
            end else if (pe_accept[row][col]) begin
              north_seen <= 1'b0;
              west_seen  <= 1'b0;
            end else begin
              if (fwd_valid[row-1][col]) north_seen <= 1'b1;
              if (fwd_valid[row][col-1]) west_seen <= 1'b1;
            end
          end
        end
      end
    end
  endgenerate

  // Done logic: Monitor bottom-right PE (PE[N-1][N-1])
  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      last_element_seen <= 1'b0;
      waiting_for_passthrough <= 1'b0;
      done_o <= 1'b0;
    end else begin
      // Step 1: Detect last_element_east_o pulse from bottom-right PE
      if (last_element_horizontal[N-1][N] && !last_element_seen) begin
        last_element_seen <= 1'b1;
        waiting_for_passthrough <= 1'b1;
      end

      // Step 2: After seeing last_element, wait for passthrough_valid_o pulse
      if (waiting_for_passthrough && passthrough_valid_o[N-1][N-1]) begin
        done_o <= 1'b1;
        waiting_for_passthrough <= 1'b0;
      end
    end
  end

endmodule

