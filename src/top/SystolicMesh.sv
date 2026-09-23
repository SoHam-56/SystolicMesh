`timescale 1ns / 100ps

module SystolicMesh #(
    parameter MATRIX_SIZE = 32,
    parameter TILE_SIZE   = 4,
    parameter DATA_WIDTH  = 32,
    parameter ROWS_MEM    = "rows.mem",
    parameter COLS_MEM    = "cols.mem"
) (
    input logic clk_i,
    input logic rstn_i,
    input logic start_matrix_mult_i,

    input logic                  north_write_enable_i,
    input logic [DATA_WIDTH-1:0] north_write_data_i,
    input logic                  north_write_reset_i,
    input logic                  west_write_enable_i,
    input logic [DATA_WIDTH-1:0] west_write_data_i,
    input logic                  west_write_reset_i,

    output logic north_queue_empty_o,
    output logic west_queue_empty_o,
    output logic matrix_mult_complete_o,
    output logic collection_complete_o,
    output logic collection_active_o,
    input  logic result_release_i,  // consumer finished reading the oldest result
    output logic input_ready_o,     // a staging bank is free for the host

    input  logic                  read_enable_i,
    input  logic [          31:0] read_addr_i,
    output logic [DATA_WIDTH-1:0] read_data_o,
    output logic                  read_valid_o
);

  localparam TILES_PER_DIM = MATRIX_SIZE / TILE_SIZE;
  localparam GLOBAL_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
  localparam TILE_ELEMENTS = TILE_SIZE * TILE_SIZE;
  localparam NUM_TILES = TILES_PER_DIM * TILES_PER_DIM;

  logic [DATA_WIDTH-1:0] mem_A[0:GLOBAL_ELEMENTS-1];
  logic [DATA_WIDTH-1:0] mem_B[0:GLOBAL_ELEMENTS-1];
  logic [$clog2(GLOBAL_ELEMENTS):0] ptr_A, ptr_B;
  logic ctrl_reset_all;  // mesh FSM re-arm, driven below

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ptr_A <= '0;
      ptr_B <= '0;
    end else begin
      // Rewind once a set is accepted; unrewound, the pointer wraps and reads back as empty.
      if (west_write_reset_i || ctrl_reset_all) ptr_A <= '0;
      else if (west_write_enable_i && ptr_A < GLOBAL_ELEMENTS) begin
        mem_A[ptr_A] <= west_write_data_i;
        ptr_A <= ptr_A + 1;
      end
      if (north_write_reset_i || ctrl_reset_all) ptr_B <= '0;
      else if (north_write_enable_i && ptr_B < GLOBAL_ELEMENTS) begin
        mem_B[ptr_B] <= north_write_data_i;
        ptr_B <= ptr_B + 1;
      end
    end
  end
  assign west_queue_empty_o  = (ptr_A == 0);
  assign north_queue_empty_o = (ptr_B == 0);

  typedef enum logic [2:0] {
    IDLE,
    RESET_SEQ,
    BROADCAST,
    FIRE_PULSE,
    WAIT_TILES,
    REDUCE_PULSE,
    WAIT_REDUCE,
    DONE
  } state_t;
  state_t current_state, next_state;

  logic loading_done;
  logic all_tiles_collected;
  logic all_reducers_done;

  logic ctrl_load_en, ctrl_fire_pulse, ctrl_reduce_pulse, ctrl_done_signal;

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] tile_col_done;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] tile_col_active;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] reducer_done;
  integer load_idx;

  assign loading_done = (load_idx >= TILE_ELEMENTS - 1);
  assign all_tiles_collected = &tile_col_done;
  assign all_reducers_done = &reducer_done;

  logic set_done;  // one cycle: this set's reduce has finished
  assign set_done = (current_state == WAIT_REDUCE) && all_reducers_done;
  assign input_ready_o = (current_state == IDLE) || (current_state == DONE);  // stub until banks exist

  always_comb begin
    next_state = current_state;
    ctrl_reset_all = 0;
    ctrl_load_en = 0;
    ctrl_fire_pulse = 0;
    ctrl_reduce_pulse = 0;
    ctrl_done_signal = 0;

    case (current_state)
      IDLE:        if (start_matrix_mult_i) next_state = RESET_SEQ;
      RESET_SEQ: begin
        ctrl_reset_all = 1;
        next_state = BROADCAST;
      end
      BROADCAST: begin
        ctrl_load_en = 1;
        if (loading_done) next_state = FIRE_PULSE;
      end
      FIRE_PULSE: begin
        ctrl_fire_pulse = 1;
        next_state = WAIT_TILES;
      end
      WAIT_TILES:  if (all_tiles_collected) next_state = REDUCE_PULSE;
      REDUCE_PULSE: begin
        ctrl_reduce_pulse = 1;
        next_state = WAIT_REDUCE;
      end
      WAIT_REDUCE: if (all_reducers_done) next_state = DONE;
      DONE: begin
        ctrl_done_signal = 1;
        if (start_matrix_mult_i) next_state = RESET_SEQ;
      end
      default:     next_state = IDLE;
    endcase
  end

  // Registered re-arm. Driving rearm_i straight from ctrl_reset_all closes a
  // combinational loop: collection_complete_o -> all_tiles_collected -> the FSM
  // that produces ctrl_reset_all. RESET_SEQ is followed by BROADCAST, so a
  // one-cycle-late clear still lands long before WAIT_TILES samples the flag.
  logic rearm_q;
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) rearm_q <= 1'b0;
    else rearm_q <= ctrl_reset_all;
  end

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] load_we_A, load_we_B;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][DATA_WIDTH-1:0] load_data_A, load_data_B;
  logic tiles_global_start;
  integer i_L, j_L, k_L, sub_r, sub_c, addr_calc;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      current_state <= IDLE;
      load_idx <= 0;
      load_we_A <= '{default: 0};
      load_we_B <= '{default: 0};
      tiles_global_start <= 0;
      matrix_mult_complete_o <= 0;
    end else begin
      current_state <= next_state;
      matrix_mult_complete_o <= ctrl_done_signal;
      tiles_global_start <= ctrl_fire_pulse;

      if (ctrl_reset_all) load_idx <= 0;
      else if (ctrl_load_en && !loading_done) load_idx <= load_idx + 1;

      load_we_A <= '{default: 0};
      load_we_B <= '{default: 0};
      if (ctrl_load_en) begin
        sub_r = load_idx / TILE_SIZE;
        sub_c = load_idx % TILE_SIZE;
        for (i_L = 0; i_L < TILES_PER_DIM; i_L++) begin
          for (k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
            addr_calc = ((i_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + ((k_L * TILE_SIZE) + sub_c);
            load_data_A[i_L][k_L] <= mem_A[addr_calc];
            load_we_A[i_L][k_L]   <= 1;
          end
        end
        for (k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
          for (j_L = 0; j_L < TILES_PER_DIM; j_L++) begin
            addr_calc = ((k_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + ((j_L * TILE_SIZE) + sub_c);
            load_data_B[k_L][j_L] <= mem_B[addr_calc];
            load_we_B[k_L][j_L]   <= 1;
          end
        end
      end
    end
  end

  logic [NUM_TILES-1:0]                 sram_we_agg;
  logic [NUM_TILES-1:0][          31:0] sram_addr_agg;
  logic [NUM_TILES-1:0][DATA_WIDTH-1:0] sram_data_agg;

  MeshOutputSram #(
      .DEPTH(GLOBAL_ELEMENTS),
      .DATA_WIDTH(DATA_WIDTH),
      .NUM_PORTS(NUM_TILES)
  ) output_mem (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .we_i(sram_we_agg),
      .waddr_i(sram_addr_agg),
      .wdata_i(sram_data_agg),
      .read_enable_i(read_enable_i),
      .read_addr_i(read_addr_i),
      .read_data_o(read_data_o),
      .read_valid_o(read_valid_o)
  );

  // reducer_done holds from the previous matmul until its reduce pulse, so an ungated
  // all_reducers_done lets a new matmul's consumer see "complete" before it has run.
  // DONE still overlaps the start cycle by one, so drop the flag while start is asserted.
  assign collection_complete_o = all_reducers_done && (current_state == DONE)
                                 && !start_matrix_mult_i;
  assign collection_active_o   = (current_state == WAIT_REDUCE);

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0]                 t_ren;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][          31:0] t_addr;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][DATA_WIDTH-1:0] t_data;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0]                 t_valid;

  genvar i, j, k;
  generate
    for (i = 0; i < TILES_PER_DIM; i++) begin : ROW
      for (j = 0; j < TILES_PER_DIM; j++) begin : COL

        localparam TILE_IDX = i * TILES_PER_DIM + j;

        AccumulationUnit #(
            .P(TILES_PER_DIM),
            .N(TILE_SIZE),
            .DATA_WIDTH(DATA_WIDTH),
            .MATRIX_WIDTH(MATRIX_SIZE),
            .TILE_ROW_OFFSET(i * TILE_SIZE),
            .TILE_COL_OFFSET(j * TILE_SIZE)
        ) acc_unit (
            .clk_i(clk_i),
            .rstn_i(rstn_i),
            .start_i(ctrl_reduce_pulse),
            .rearm_i(rearm_q),
            .tile_data_i(t_data[i][j]),
            .tile_valid_i(t_valid[i][j]),
            .tile_ren_o(t_ren[i][j]),
            .tile_addr_o(t_addr[i][j]),

            .write_en_o  (sram_we_agg[TILE_IDX]),
            .write_addr_o(sram_addr_agg[TILE_IDX]),
            .write_data_o(sram_data_agg[TILE_IDX]),

            .done_o(reducer_done[i][j])
        );

        for (k = 0; k < TILES_PER_DIM; k++) begin : DEPTH
          SystolicArray #(
              .N(TILE_SIZE),
              .DATA_WIDTH(DATA_WIDTH),
              .ROWS(ROWS_MEM),
              .COLS(COLS_MEM)
          ) tile (
              .clk_i(clk_i),
              .rstn_i(rstn_i),
              .start_matrix_mult_i(tiles_global_start),
              .rearm_i(rearm_q),
              .west_write_enable_i(load_we_A[i][k]),
              .west_write_data_i(load_data_A[i][k]),
              .west_write_reset_i(ctrl_reset_all),
              .north_write_enable_i(load_we_B[k][j]),
              .north_write_data_i(load_data_B[k][j]),
              .north_write_reset_i(ctrl_reset_all),
              .collection_complete_o(tile_col_done[i][j][k]),
              .collection_active_o(tile_col_active[i][j][k]),
              .matrix_mult_complete_o(),
              .north_queue_empty_o(),
              .west_queue_empty_o(),
              .read_enable_i(t_ren[i][j][k]),
              .read_addr_i(t_addr[i][j][k]),
              .read_data_o(t_data[i][j][k]),
              .read_valid_o(t_valid[i][j][k])
          );
        end
      end
    end
  endgenerate

endmodule
