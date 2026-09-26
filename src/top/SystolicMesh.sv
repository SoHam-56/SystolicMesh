`timescale 1ns / 100ps

module SystolicMesh #(
    parameter MATRIX_SIZE = 32,
    parameter TILE_SIZE   = 4,
    parameter DATA_WIDTH  = 32,
    parameter WIDE_READ   = 1,  // words per wide result read, one per consumer lane
    parameter HOST_WORDS  = MATRIX_SIZE,  // words per host write, one matrix row; must divide MATRIX_SIZE*MATRIX_SIZE
    parameter COLLAPSE_K  = 1,  // 1: one full-depth tile per output tile, N^2 PEs and no reduce; 0: depth slices and the reduce tree
    parameter RESULT_BANKS = 4,  // results held for the consumer: four cover the reduce latency and the consumer's read
    parameter ACC_BANKS    = 4,  // partial-sum banks per PE: a bank returns about 3K cycles after its set starts, so 4 keep K per set
    parameter WC_TILES     = 128,  // weight cache: N x N tiles of B, in two regions (tile MSB) so one fills while the other is read
    parameter WCTW         = $clog2(WC_TILES),
    parameter WCAW         = $clog2(WC_TILES * MATRIX_SIZE * MATRIX_SIZE)
) (
    input logic clk_i,
    input logic rstn_i,
    input logic start_matrix_mult_i,
    input logic                                  bias_valid_i,  // with the start: add bias_i[c] to every element of column c
    input logic [MATRIX_SIZE-1:0][DATA_WIDTH-1:0] bias_i,
    input logic                                  weight_cached_i,  // with the start: B is cache tile weight_tile_i, the host sends only A
    input logic [WCTW-1:0]                       weight_tile_i,
    input logic                                  wc_write_enable_i,  // cache write of north_write_data_i at word wc_write_addr_i
    input logic [WCAW-1:0]                       wc_write_addr_i,
    output logic [1:0]                           wc_region_busy_o,   // a started, not yet broadcast set reads this region

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
  logic [DATA_WIDTH-1:0] wcache[WC_TILES*GLOBAL_ELEMENTS];
  logic [1:0] in_cached;  // per staging bank: B comes from the cache
  logic [WCTW-1:0] in_tile[2];  // and from this tile
  logic [$clog2(GLOBAL_ELEMENTS):0] ptr_A, ptr_B;
  initial if ((GLOBAL_ELEMENTS % HOST_WORDS) != 0) $error("SystolicMesh: HOST_WORDS (%0d) must divide %0d", HOST_WORDS, GLOBAL_ELEMENTS);
  logic [1:0] in_full;  // per staging bank: a started set not yet broadcast
  logic in_wr, in_rd;  // bank the host writes, bank BROADCAST reads
  logic start_accept, bcast_release;
  assign input_ready_o = !in_full[in_wr];
  assign start_accept  = start_matrix_mult_i && input_ready_o;
  logic west_wr_ok, north_wr_ok;
  assign west_wr_ok  = west_write_enable_i && input_ready_o && !west_write_reset_i && ptr_A < GLOBAL_ELEMENTS;
  assign north_wr_ok = north_write_enable_i && input_ready_o && !north_write_reset_i && ptr_B < GLOBAL_ELEMENTS;
  localparam int RBW = $clog2(RESULT_BANKS);
  logic [RESULT_BANKS-1:0] out_full;  // per result bank: holds a finished, unreleased result
  logic loading_done;
  logic [RBW-1:0] out_wr, out_rd;  // bank the reducers write, bank the consumer reads

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      ptr_A   <= '0;
      ptr_B   <= '0;
      in_full <= '0;
      in_wr   <= 1'b0;
      in_rd   <= 1'b0;
      in_cached <= '0;
      in_tile[0] <= '0;
      in_tile[1] <= '0;
    end else begin
      // A write in the start cycle is the set's last row and lands before the bank switches.
      if (west_wr_ok)
        for (int c = 0; c < HOST_WORDS; c++) mem_A[int'(in_wr)*GLOBAL_ELEMENTS+int'(ptr_A)+c] <= west_write_data_i[c];
      if (north_wr_ok)
        for (int c = 0; c < HOST_WORDS; c++) mem_B[int'(in_wr)*GLOBAL_ELEMENTS+int'(ptr_B)+c] <= north_write_data_i[c];
      // Rewind as each set is accepted; unrewound, the pointer wraps and reads back as empty.
      if (west_write_reset_i || start_accept) ptr_A <= '0;
      else if (west_wr_ok) ptr_A <= ptr_A + HOST_WORDS;
      if (north_write_reset_i || start_accept) ptr_B <= '0;
      else if (north_wr_ok) ptr_B <= ptr_B + HOST_WORDS;
      if (start_accept) begin
        in_full[in_wr] <= 1'b1;
        in_cached[in_wr] <= weight_cached_i;
        in_tile[in_wr] <= weight_tile_i;
        in_wr <= ~in_wr;
      end
      if (bcast_release) begin
        in_full[in_rd] <= 1'b0;
        in_rd <= ~in_rd;
      end
    end
  end
  assign west_queue_empty_o  = (ptr_A == 0);

  // ── Weight cache: written from the north bus, read by the broadcaster for cached sets ──
  always_ff @(posedge clk_i) begin
    if (wc_write_enable_i)
      for (int c = 0; c < HOST_WORDS; c++) wcache[int'(wc_write_addr_i)+c] <= north_write_data_i[c];
  end
  always_comb begin
    wc_region_busy_o = '0;
    for (int b = 0; b < 2; b++)
      if (in_full[b] && in_cached[b]) wc_region_busy_o[in_tile[b][WCTW-1]] = 1'b1;
  end
  function automatic logic [DATA_WIDTH-1:0] b_word(input int addr);
    return in_cached[in_rd] ? wcache[int'(in_tile[in_rd])*GLOBAL_ELEMENTS+addr] : mem_B[int'(in_rd)*GLOBAL_ELEMENTS+addr];
  endfunction
  assign north_queue_empty_o = (ptr_B == 0);

  // ── Broadcast: copy a full staging bank into every array's free operand bank, one tile row per cycle ──
  localparam int AK = COLLAPSE_K ? MATRIX_SIZE : TILE_SIZE;  // depth of each array's product
  localparam int U = (AK < 6) ? AK : 6;  // partial sums per array pixel
  localparam int RPU = RP * U;  // partials the reducer sums per pixel
  localparam int BIAS_Q = 1 << $clog2(4 + ACC_BANKS + 1);  // sets between start and reduce: staging, operand and partial-sum banks

  typedef enum logic [1:0] {
    B_IDLE,
    B_LOAD,
    B_COMMIT
  } bstate_t;
  bstate_t bstate;

  logic arrays_load_ready;  // every array has a free operand bank
  logic arrays_final;  // every array holds a final unread set
  logic arrays_next_final;  // and the set after it is final too
  logic arrays_busy;
  logic reducers_ready, reducers_busy, reducers_read_done, reducers_written;
  logic ctrl_load_en, commit_q, set_launch, reduce_start;
  integer load_idx;

  assign loading_done  = (load_idx >= TILE_SIZE - 1);  // one tile row per cycle
  assign ctrl_load_en  = (bstate == B_LOAD);
  assign bcast_release = (bstate == B_LOAD) && loading_done;  // staging bank copied into the arrays
  assign set_launch    = (bstate == B_IDLE) && in_full[in_rd] && arrays_load_ready;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      bstate   <= B_IDLE;
      load_idx <= 0;
      commit_q <= 1'b0;
    end else begin
      commit_q <= bcast_release;  // lands with the last registered row write
      case (bstate)
        B_IDLE:
        if (set_launch) begin
          bstate   <= B_LOAD;
          load_idx <= 0;
        end
        B_LOAD: begin
          if (loading_done) bstate <= B_COMMIT;
          else load_idx <= load_idx + 1;
        end
        B_COMMIT: bstate <= B_IDLE;  // the arrays switch operand bank before the next launch looks
        default: bstate <= B_IDLE;
      endcase
    end
  end

  // ── Result banks: FREE, WRITING from reduce start, FULL once written, FREE again on release ──
  typedef enum logic [1:0] {
    R_FREE,
    R_WRITING,
    R_FULL
  } rstate_t;
  rstate_t out_state[RESULT_BANKS];
  logic set_done;  // one cycle: a set's last result was written
  logic [RBW-1:0] wr_bank_done;  // bank the next written set belongs to: sets finish in order
  function automatic logic [RBW-1:0] next_bank(input logic [RBW-1:0] b);
    return (b == RBW'(RESULT_BANKS - 1)) ? '0 : b + 1'b1;
  endfunction
  for (genvar b = 0; b < RESULT_BANKS; b++) begin : OUT_FULL
    assign out_full[b] = (out_state[b] == R_FULL);
  end
  // Idle reducers take the oldest final set; reducers reading their last pixel take the next one, as the oldest is released.
  assign reduce_start = reducers_ready && (out_state[out_wr] == R_FREE) &&
                        (reducers_read_done ? arrays_next_final : arrays_final);
  assign set_done     = reducers_written;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int b = 0; b < RESULT_BANKS; b++) out_state[b] <= R_FREE;
      out_wr <= '0;
      out_rd <= '0;
      wr_bank_done <= '0;
      matrix_mult_complete_o <= 1'b0;
    end else begin
      matrix_mult_complete_o <= set_done;
      if (reduce_start) begin
        out_state[out_wr] <= R_WRITING;
        out_wr <= next_bank(out_wr);
      end
      if (set_done) begin
        out_state[wr_bank_done] <= R_FULL;
        wr_bank_done <= next_bank(wr_bank_done);
      end
      if (result_release_i && out_full[out_rd]) begin
        out_state[out_rd] <= R_FREE;
        out_rd <= next_bank(out_rd);
      end
    end
  end

  // ── Bias queue: sets reach the reducers in the order they were started ──
  logic [DATA_WIDTH-1:0] bias_q[BIAS_Q][MATRIX_SIZE];
  logic [BIAS_Q-1:0] bias_qv;
  logic [$clog2(BIAS_Q)-1:0] bq_wr, bq_rd;
  logic [$clog2(BIAS_Q):0] bq_n;
  logic [MATRIX_SIZE-1:0][DATA_WIDTH-1:0] red_bias;  // bias of the set the reducers are starting
  always_comb begin
    for (int c = 0; c < MATRIX_SIZE; c++) red_bias[c] = bias_qv[bq_rd] ? bias_q[bq_rd][c] : '0;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      bq_wr   <= '0;
      bq_rd   <= '0;
      bq_n    <= '0;
      bias_qv <= '0;
    end else begin
      if (start_accept) begin
        for (int c = 0; c < MATRIX_SIZE; c++) bias_q[bq_wr][c] <= bias_i[c];
        bias_qv[bq_wr] <= bias_valid_i;
        bq_wr <= bq_wr + 1'b1;
      end
      if (reduce_start) bq_rd <= bq_rd + 1'b1;
      bq_n <= bq_n + (start_accept ? 1'b1 : 1'b0) - (reduce_start ? 1'b1 : 1'b0);
    end
  end

  logic mesh_busy;  // a set is somewhere between staging and a written result; for testbenches
  assign mesh_busy = (bstate != B_IDLE) || arrays_busy || reducers_busy;

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] load_we_A, load_we_B;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][LW-1:0][DATA_WIDTH-1:0] load_data_A, load_data_B;
  integer i_L, j_L, k_L, sub_r, sub_c, addr_calc, w_L, rr_L;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      load_we_A <= '{default: 0};
      load_we_B <= '{default: 0};
    end else begin
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
              load_data_B[0][j_L][w_L] <= b_word(addr_calc);
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
                load_data_B[k_L][j_L][sub_c] <= b_word(addr_calc);
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
    for (int p = 0; p < NUM_TILES; p++) sram_addr_bank[p] = sram_addr_agg[p];  // each reducer adds its own bank offset

  localparam int WIDE_STRIDE = GLOBAL_ELEMENTS / WIDE_READ;
  logic [WIDE_READ-1:0][31:0] wide_addr;
  always_comb
    for (int k = 0; k < WIDE_READ; k++)
      wide_addr[k] = int'(out_rd) * GLOBAL_ELEMENTS + k * WIDE_STRIDE + wide_read_index_i;

  initial
    if (GLOBAL_ELEMENTS % WIDE_READ != 0)
      $error("SystolicMesh: WIDE_READ (%0d) must divide N*N (%0d)", WIDE_READ, GLOBAL_ELEMENTS);

  MeshOutputSram #(
      .DEPTH(RESULT_BANKS * GLOBAL_ELEMENTS),
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
      .read_addr_i(read_addr_i + int'(out_rd) * GLOBAL_ELEMENTS),
      .read_data_o(read_data_o),
      .read_valid_o(read_valid_o),
      .wide_enable_i(wide_read_enable_i && wide_read_index_i < WIDE_STRIDE),
      .wide_addr_i(wide_addr),
      .wide_data_o(wide_read_data_o),
      .wide_valid_o(wide_read_valid_o)
  );

  assign collection_complete_o = out_full[out_rd];  // cleared by release, never sticky
  assign collection_active_o   = reducers_busy;

  // Per output tile: U partials from each depth slice, flattened for its reducer.
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][RPU-1:0][DATA_WIDTH-1:0] t_data;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] t_ren;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][DATA_WIDTH-1:0] t_bias;  // the bias of the pixel being summed
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][$clog2(TILE_ELEMENTS)-1:0] t_addr;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] r_ready, r_busy, r_read_done, r_written;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] a_ready, a_final, a_next, a_busy;

  // Every array and reducer runs in lockstep; the mesh acts on the AND (or OR) of them all.
  always_comb begin
    arrays_load_ready = 1'b1;
    arrays_final = 1'b1;
    arrays_next_final = 1'b1;
    arrays_busy = 1'b0;
    reducers_ready = 1'b1;
    reducers_busy = 1'b0;
    reducers_read_done = 1'b1;
    reducers_written = 1'b1;
    for (int a = 0; a < TILES_PER_DIM; a++)
      for (int b = 0; b < TILES_PER_DIM; b++) begin
        reducers_ready &= r_ready[a][b];
        reducers_busy |= r_busy[a][b];
        reducers_read_done &= r_read_done[a][b];
        reducers_written &= r_written[a][b];
        for (int c = 0; c < TILES_PER_DIM; c++) begin
          arrays_load_ready &= a_ready[a][b][c];
          arrays_final &= a_final[a][b][c];
          arrays_next_final &= a_next[a][b][c];
          arrays_busy |= a_busy[a][b][c];
        end
      end
  end

  genvar i, j, k;
  generate
    for (i = 0; i < TILES_PER_DIM; i++) begin : ROW
      for (j = 0; j < TILES_PER_DIM; j++) begin : COL

        localparam TILE_IDX = i * TILES_PER_DIM + j;

        AccumulationUnit #(
            .P(RPU + 1),  // the partials and the bias
            .RESULT_BANKS(RESULT_BANKS),
            .N(TILE_SIZE),
            .DATA_WIDTH(DATA_WIDTH),
            .MATRIX_WIDTH(MATRIX_SIZE),
            .TILE_ROW_OFFSET(i * TILE_SIZE),
            .TILE_COL_OFFSET(j * TILE_SIZE)
        ) acc_unit (
            .clk_i       (clk_i),
            .rstn_i      (rstn_i),
            .start_i     (reduce_start),
            .out_bank_i  (out_wr),
            .tile_data_i ({t_bias[i][j], t_data[i][j]}),
            .bias_i      (red_bias[j*TILE_SIZE+:TILE_SIZE]),
            .rd_en_o     (t_ren[i][j]),
            .rd_addr_o   (t_addr[i][j]),
            .read_done_o (r_read_done[i][j]),
            .ready_o     (r_ready[i][j]),
            .write_en_o  (sram_we_agg[TILE_IDX]),
            .write_addr_o(sram_addr_agg[TILE_IDX]),
            .write_data_o(sram_data_agg[TILE_IDX]),
            .written_o   (r_written[i][j]),
            .bias_word_o (t_bias[i][j]),
            .busy_o      (r_busy[i][j])
        );

        for (k = 0; k < TILES_PER_DIM; k++) begin : DEPTH
          if (COLLAPSE_K && k > 0) begin : UNUSED
            // Collapsed: only depth slot 0 exists; the rest read as ready, final and idle.
            assign a_ready[i][j][k] = 1'b1;
            assign a_final[i][j][k] = 1'b1;
            assign a_next[i][j][k]  = 1'b1;
            assign a_busy[i][j][k]  = 1'b0;
          end else begin : S
            logic [U-1:0][DATA_WIDTH-1:0] rd;
            SystolicArray #(
                .N(TILE_SIZE),
                .K(AK),
                .DATA_WIDTH(DATA_WIDTH),
                .WEST_WORDS(LW),
                .NORTH_WORDS(LW),
                .U(U),
                .BANKS(ACC_BANKS)
            ) tile (
                .clk_i(clk_i),
                .rstn_i(rstn_i),
                .west_write_enable_i(load_we_A[i][k]),
                .west_write_data_i(load_data_A[i][k]),
                .north_write_enable_i(load_we_B[k][j]),
                .north_write_data_i(load_data_B[k][j]),
                .commit_i(commit_q),
                .load_ready_o(a_ready[i][j][k]),
                .set_final_o(a_final[i][j][k]),
                .next_final_o(a_next[i][j][k]),
                .read_enable_i(t_ren[i][j]),
                .read_addr_i(t_addr[i][j]),
                .read_data_o(rd),
                .read_valid_o(),
                .release_i(r_read_done[i][j]),
                .busy_o(a_busy[i][j][k])
            );
            for (genvar u = 0; u < U; u++) begin : PART
              assign t_data[i][j][k*U+u] = rd[u];
            end
          end
        end
      end
    end
  endgenerate

`ifndef SYNTHESIS
  // Handshake invariants; live only with --assert.
  a_result_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) reduce_start |-> out_state[out_wr] == R_FREE)
    else $error("SystolicMesh: a reduce started into a result bank that is not free");
  a_written_in_order: assert property (@(posedge clk_i) disable iff (!rstn_i) set_done |-> out_state[wr_bank_done] == R_WRITING)
    else $error("SystolicMesh: a set finished writing into a bank that was not being written");
  a_bias_queue_room: assert property (@(posedge clk_i) disable iff (!rstn_i) start_accept |-> bq_n < BIAS_Q)
    else $error("SystolicMesh: bias queue overflow");
  a_bias_queue_held: assert property (@(posedge clk_i) disable iff (!rstn_i) reduce_start |-> bq_n != 0)
    else $error("SystolicMesh: a reduce started with no bias queued");
  a_wc_bus: assert property (@(posedge clk_i) disable iff (!rstn_i) !(wc_write_enable_i && north_write_enable_i))
    else $error("SystolicMesh: a cache write and a B write share the north bus in one cycle");
  a_wc_region_free: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                     wc_write_enable_i |-> !wc_region_busy_o[wc_write_addr_i[WCAW-1]])
    else $error("SystolicMesh: cache write into a region a staged set still reads");
  a_staging_bank_full: assert property (@(posedge clk_i) disable iff (!rstn_i) bcast_release |-> in_full[in_rd])
    else $error("SystolicMesh: BROADCAST copied an empty staging bank");
  a_arrays_ready_on_launch: assert property (@(posedge clk_i) disable iff (!rstn_i) (load_we_A != '0) |-> arrays_load_ready)
    else $error("SystolicMesh: broadcast wrote an array whose operand bank was not free");
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
