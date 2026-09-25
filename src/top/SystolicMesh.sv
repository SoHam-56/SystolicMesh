`timescale 1ns / 100ps

module SystolicMesh #(
    parameter MATRIX_SIZE = 32,
    parameter TILE_SIZE   = 4,
    parameter DATA_WIDTH  = 32,
    parameter WIDE_READ   = 1,  // words per wide result read, one per consumer lane
    parameter HOST_WORDS  = MATRIX_SIZE,  // words per host write, one matrix row; must divide MATRIX_SIZE*MATRIX_SIZE
    parameter COLLAPSE_K  = 1   // 1: one full-depth tile per output tile, N^2 PEs and no reduce; 0: depth slices and the reduce tree
) (
    input logic clk_i,
    input logic rstn_i,
    input logic start_matrix_mult_i,

    input logic                  north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                  north_write_reset_i,
    input logic                  west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
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
    output logic                  read_valid_o,

    // Wide read: word k is element k * (N*N / WIDE_READ) + wide_read_index_i of the oldest result.
    input  logic                                 wide_read_enable_i,
    input  logic [                         31:0] wide_read_index_i,
    output logic [WIDE_READ-1:0][DATA_WIDTH-1:0] wide_read_data_o,
    output logic                                 wide_read_valid_o
);

  localparam TILES_PER_DIM = MATRIX_SIZE / TILE_SIZE;
  localparam GLOBAL_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE;
  localparam TILE_ELEMENTS = TILE_SIZE * TILE_SIZE;
  localparam NUM_TILES = TILES_PER_DIM * TILES_PER_DIM;
  localparam RP = COLLAPSE_K ? 1 : TILES_PER_DIM;  // partial tiles per output tile
  localparam LW = COLLAPSE_K ? MATRIX_SIZE : TILE_SIZE;  // words per broadcast write

  logic [DATA_WIDTH-1:0] mem_A[0:2*GLOBAL_ELEMENTS-1];
  logic [DATA_WIDTH-1:0] mem_B[0:2*GLOBAL_ELEMENTS-1];
  logic [$clog2(GLOBAL_ELEMENTS):0] ptr_A, ptr_B;
  initial if ((GLOBAL_ELEMENTS % HOST_WORDS) != 0) $error("SystolicMesh: HOST_WORDS (%0d) must divide %0d", HOST_WORDS, GLOBAL_ELEMENTS);
  logic [1:0] in_full;  // per staging bank: a started set not yet broadcast
  logic in_wr, in_rd;  // bank the host writes, bank BROADCAST reads
  logic start_accept, bcast_release;
  assign input_ready_o = !in_full[in_wr];
  assign start_accept  = start_matrix_mult_i && input_ready_o;
  logic ctrl_reset_all;  // mesh FSM re-arm, driven below
  logic [1:0] out_full;  // per result bank: holds a finished, unreleased result
  logic out_wr, out_rd;  // bank the reducers write, bank the consumer reads

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ptr_A   <= '0;
      ptr_B   <= '0;
      in_full <= '0;
      in_wr   <= 1'b0;
      in_rd   <= 1'b0;
    end else begin
      // Rewind as each set is accepted; unrewound, the pointer wraps and reads back as empty.
      if (west_write_reset_i || start_accept) ptr_A <= '0;
      else if (west_write_enable_i && input_ready_o && ptr_A < GLOBAL_ELEMENTS) begin
        for (int c = 0; c < HOST_WORDS; c++) mem_A[int'(in_wr)*GLOBAL_ELEMENTS+int'(ptr_A)+c] <= west_write_data_i[c];
        ptr_A <= ptr_A + HOST_WORDS;
      end
      if (north_write_reset_i || start_accept) ptr_B <= '0;
      else if (north_write_enable_i && input_ready_o && ptr_B < GLOBAL_ELEMENTS) begin
        for (int c = 0; c < HOST_WORDS; c++) mem_B[int'(in_wr)*GLOBAL_ELEMENTS+int'(ptr_B)+c] <= north_write_data_i[c];
        ptr_B <= ptr_B + HOST_WORDS;
      end
      if (start_accept) begin
        in_full[in_wr] <= 1'b1;
        in_wr <= ~in_wr;
      end
      if (bcast_release) begin
        in_full[in_rd] <= 1'b0;
        in_rd <= ~in_rd;
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

  assign loading_done = (load_idx >= TILE_SIZE - 1);  // one tile row per cycle
  assign bcast_release = (current_state == BROADCAST) && loading_done;  // bank copied into the tiles
  assign all_tiles_collected = &tile_col_done;
  assign all_reducers_done = &reducer_done;

  logic set_done;  // one cycle: this set's reduce has finished
  assign set_done = (current_state == WAIT_REDUCE) && all_reducers_done;

  always_comb begin
    next_state = current_state;
    ctrl_reset_all = 0;
    ctrl_load_en = 0;
    ctrl_fire_pulse = 0;
    ctrl_reduce_pulse = 0;
    ctrl_done_signal = 0;

    case (current_state)
      IDLE:        if (in_full[in_rd] && !out_full[out_wr]) next_state = RESET_SEQ;
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
        if (in_full[in_rd] && !out_full[out_wr]) next_state = RESET_SEQ;
      end
      default:     next_state = IDLE;
    endcase
  end

  // Result banks: set by a finished reduce, cleared by the consumer's release.
  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      out_full <= '0;
      out_wr   <= 1'b0;
      out_rd   <= 1'b0;
    end else begin
      if (set_done) begin
        out_full[out_wr] <= 1'b1;
        out_wr <= ~out_wr;
      end
      if (result_release_i && out_full[out_rd]) begin
        out_full[out_rd] <= 1'b0;
        out_rd <= ~out_rd;
      end
    end
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
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][LW-1:0][DATA_WIDTH-1:0] load_data_A, load_data_B;
  logic tiles_global_start;
  integer i_L, j_L, k_L, sub_r, sub_c, addr_calc, w_L, rr_L;

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
        sub_r = load_idx;  // tile row copied this cycle
        if (COLLAPSE_K) begin
          // Tile row i takes row sub_r of its T x N slab of A; tile column j takes N/T rows of its N x T slab of B.
          for (i_L = 0; i_L < TILES_PER_DIM; i_L++) begin
            for (sub_c = 0; sub_c < MATRIX_SIZE; sub_c++) begin
              addr_calc = ((i_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + sub_c;
              load_data_A[i_L][0][sub_c] <= mem_A[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
            end
            load_we_A[i_L][0] <= 1;
          end
          for (j_L = 0; j_L < TILES_PER_DIM; j_L++) begin
            for (w_L = 0; w_L < MATRIX_SIZE; w_L++) begin
              rr_L = sub_r * TILES_PER_DIM + w_L / TILE_SIZE;
              addr_calc = rr_L * MATRIX_SIZE + j_L * TILE_SIZE + w_L % TILE_SIZE;
              load_data_B[0][j_L][w_L] <= mem_B[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
            end
            load_we_B[0][j_L] <= 1;
          end
        end else begin
        for (i_L = 0; i_L < TILES_PER_DIM; i_L++) begin
          for (k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
            for (sub_c = 0; sub_c < TILE_SIZE; sub_c++) begin
              addr_calc = ((i_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + ((k_L * TILE_SIZE) + sub_c);
              load_data_A[i_L][k_L][sub_c] <= mem_A[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
            end
            load_we_A[i_L][k_L] <= 1;
          end
        end
        for (k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
          for (j_L = 0; j_L < TILES_PER_DIM; j_L++) begin
            for (sub_c = 0; sub_c < TILE_SIZE; sub_c++) begin
              addr_calc = ((k_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + ((j_L * TILE_SIZE) + sub_c);
              load_data_B[k_L][j_L][sub_c] <= mem_B[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
            end
            load_we_B[k_L][j_L] <= 1;
          end
        end
        end
      end
    end
  end

  logic [NUM_TILES-1:0]                 sram_we_agg;
  logic [NUM_TILES-1:0][          31:0] sram_addr_agg;
  logic [NUM_TILES-1:0][DATA_WIDTH-1:0] sram_data_agg;
  logic [NUM_TILES-1:0][          31:0] sram_addr_bank;

  always_comb
    for (int p = 0; p < NUM_TILES; p++)
      sram_addr_bank[p] = sram_addr_agg[p] + (out_wr ? GLOBAL_ELEMENTS : 0);

  localparam int WIDE_STRIDE = GLOBAL_ELEMENTS / WIDE_READ;
  logic [WIDE_READ-1:0][31:0] wide_addr;
  always_comb
    for (int k = 0; k < WIDE_READ; k++)
      wide_addr[k] = (out_rd ? GLOBAL_ELEMENTS : 0) + k * WIDE_STRIDE + wide_read_index_i;

  initial
    if (GLOBAL_ELEMENTS % WIDE_READ != 0)
      $error("SystolicMesh: WIDE_READ (%0d) must divide N*N (%0d)", WIDE_READ, GLOBAL_ELEMENTS);

  MeshOutputSram #(
      .DEPTH(2 * GLOBAL_ELEMENTS),
      .DATA_WIDTH(DATA_WIDTH),
      .NUM_PORTS(NUM_TILES),
      .WIDE(WIDE_READ)
  ) output_mem (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .we_i(sram_we_agg),
      .waddr_i(sram_addr_bank),
      .wdata_i(sram_data_agg),
      .read_enable_i(read_enable_i && read_addr_i < GLOBAL_ELEMENTS),
      .read_addr_i(read_addr_i + (out_rd ? GLOBAL_ELEMENTS : 0)),
      .read_data_o(read_data_o),
      .read_valid_o(read_valid_o),
      .wide_enable_i(wide_read_enable_i && wide_read_index_i < WIDE_STRIDE),
      .wide_addr_i(wide_addr),
      .wide_data_o(wide_read_data_o),
      .wide_valid_o(wide_read_valid_o)
  );

  assign collection_complete_o = out_full[out_rd];  // cleared by release, never sticky
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
            .P(RP),
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
            .tile_data_i(t_data[i][j][RP-1:0]),
            .tile_valid_i(t_valid[i][j][RP-1:0]),
            .tile_ren_o(t_ren[i][j][RP-1:0]),
            .tile_addr_o(t_addr[i][j][RP-1:0]),

            .write_en_o  (sram_we_agg[TILE_IDX]),
            .write_addr_o(sram_addr_agg[TILE_IDX]),
            .write_data_o(sram_data_agg[TILE_IDX]),

            .done_o(reducer_done[i][j])
        );

        for (k = 0; k < TILES_PER_DIM; k++) begin : DEPTH
          if (COLLAPSE_K && k > 0) begin : UNUSED
            // Collapsed: only depth slot 0 exists; the rest read as finished and empty.
            assign tile_col_done[i][j][k] = 1'b1;
            assign tile_col_active[i][j][k] = 1'b0;
            assign t_data[i][j][k] = '0;
            assign t_valid[i][j][k] = 1'b0;
          end else begin : S
            SystolicArray #(
                .N(TILE_SIZE),
                .K(COLLAPSE_K ? MATRIX_SIZE : TILE_SIZE),
                .DATA_WIDTH(DATA_WIDTH),
                .WEST_WORDS(LW),
                .NORTH_WORDS(LW)
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
    end
  endgenerate

`ifndef SYNTHESIS
  // Handshake invariants; live only with --assert.
  a_result_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) set_done |-> !out_full[out_wr])
    else $error("SystolicMesh: a reduce finished into a full result bank");
  a_staging_bank_full: assert property (@(posedge clk_i) disable iff (!rstn_i) bcast_release |-> in_full[in_rd])
    else $error("SystolicMesh: BROADCAST copied an empty staging bank");
  a_launch_ready: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                   ctrl_reset_all |-> (in_full[in_rd] && !out_full[out_wr]))
    else $error("SystolicMesh: launched without a full staging bank and a free result bank");
  a_read_outstanding: assert property (@(posedge clk_i) disable iff (!rstn_i) read_enable_i |-> out_full[out_rd])
    else $error("SystolicMesh: result read with no result outstanding");
  a_wide_read_outstanding: assert property (@(posedge clk_i) disable iff (!rstn_i) wide_read_enable_i |-> out_full[out_rd])
    else $error("SystolicMesh: wide result read with no result outstanding");
`ifdef ASSERT_SELFTEST
  a_selftest: assert property (@(posedge clk_i) disable iff (!rstn_i) 1'b0)
    else $error("SystolicMesh: assertion self-test fired, so assertions are live");
`endif
`endif

endmodule
