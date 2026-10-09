`timescale 1ns / 100ps

module SystolicMesh #(
    parameter MATRIX_SIZE = 32,
    parameter TILE_SIZE   = 4,
    parameter EXP_W       = 8,   // the build's format: fp32 8/23 by default, bf16 8/7, int8 0/7
    parameter MAN_W       = 23,
    parameter DATA_WIDTH  = 1 + EXP_W + MAN_W,  // operands: host writes, staging banks, weight cache
    parameter ACC_W       = sienna_fmt_pkg::acc_w(EXP_W, MAN_W),  // sums and bias: int32 in int8, fp32 in every float format
    parameter OUT_W       = sienna_fmt_pkg::out_w(EXP_W, MAN_W),  // result words: int32 in int8, the format's own width in floats
    parameter WIDE_READ   = 1,  // words per wide result read, one per consumer lane
    parameter HOST_WORDS  = MATRIX_SIZE,  // words per host write, one matrix row; must divide MATRIX_SIZE*MATRIX_SIZE
    parameter COLLAPSE_K  = 1,  // 1: one full-depth tile per output tile, N^2 PEs and no reduce; 0: depth slices and the reduce tree
    parameter RESULT_BANKS = 4,  // results held for the consumer: four cover the reduce latency and the consumer's read
    parameter ACC_BANKS    = 4,  // partial-sum banks per PE: a bank returns about 3K cycles after its set starts, so 4 keep K per set
    parameter WC_TILES     = 128,  // weight cache: N x N tiles of B, in two regions (tile MSB) so one fills while the other is read
    parameter WCTW         = $clog2(WC_TILES),
    parameter WCAW         = $clog2(WC_TILES * MATRIX_SIZE * MATRIX_SIZE),
    parameter int RES_MAX  = MATRIX_SIZE * MATRIX_SIZE / WIDE_READ,  // the most result credits the consumer may advertise
    parameter int RES_CRW  = 1  // result.credit width
) (
    input logic clk_i,
    input logic rstn_i,
    credit_link_if.consumer staging,  // L1: a put per set, data {wc_last, weight_tile, weight_cached, pack_shift[2:0], bias_valid, partial}
    input logic [MATRIX_SIZE-1:0][ACC_W-1:0]      bias_i,  // with the put: add bias_i[c] to every element of column c (float builds: fp32 bits, the caller widens)
    credit_link_if.consumer wc_region[2],  // L2: a put opens a fill of that region; its credit returns once the fill's last set is broadcast
    input logic                                  wc_write_enable_i,  // cache write of north_write_data_i at word wc_write_addr_i
    input logic [WCAW-1:0]                       wc_write_addr_i,

    input logic                  north_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] north_write_data_i,
    input logic                  north_write_reset_i,
    input logic                  west_write_enable_i,
    input logic [HOST_WORDS-1:0][DATA_WIDTH-1:0] west_write_data_i,
    input logic                  west_write_reset_i,

    output logic north_queue_empty_o,
    output logic west_queue_empty_o,
    output logic matrix_mult_complete_o,

    credit_link_if.producer result  // L3: N*N/WIDE_READ beats per result, data {packed, last, first, WIDE_READ words}
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
  logic [1:0] in_fresh, in_more;  // per staging bank: the set starts a sum; the next set continues it
  logic last_partial;  // the previous accepted set continues into the next one
  logic [WCTW-1:0] in_tile[2];  // and from this tile
  logic [2:0] in_pack[2];  // per staging bank: the set's pack shift
  logic [1:0] in_last;  // per staging bank: the set is the last to read its cache region before the region's next fill
  logic [$clog2(GLOBAL_ELEMENTS):0] ptr_A, ptr_B;
`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial if ((GLOBAL_ELEMENTS % HOST_WORDS) != 0) $error("SystolicMesh: HOST_WORDS (%0d) must divide %0d", HOST_WORDS, GLOBAL_ELEMENTS);
  initial if (DATA_WIDTH != 1 + EXP_W + MAN_W) $error("SystolicMesh: DATA_WIDTH %0d is not 1 + EXP_W + MAN_W", DATA_WIDTH);
`endif
  if (OUT_W != sienna_fmt_pkg::out_w(EXP_W, MAN_W)) begin : G_BAD_OUT_W  // the result banks and link would truncate or pad silently
    $fatal(1, "SystolicMesh: OUT_W=%0d is not sienna_fmt_pkg::out_w(%0d, %0d)", OUT_W, EXP_W, MAN_W);
  end
  logic [1:0] in_full;  // per staging bank: a started set not yet broadcast
  logic in_wr, in_rd;  // bank the host writes, bank BROADCAST reads
  logic in_room;  // the bank the host writes is free; a host holding a staging credit always finds it so
  logic start_accept, bcast_release;

  // ── L1: one staging credit per bank, both advertised after reset, then one per broadcast ──
  localparam int STG_W = WCTW + 7;
  logic stg_partial, stg_bias_v, stg_cached, stg_last;  // the put's sideband, fields as in the staging port comment
  logic [2:0] stg_pack;
  logic [WCTW-1:0] stg_tile;
  assign {stg_last, stg_tile, stg_cached, stg_pack, stg_bias_v, stg_partial} = staging.data[STG_W-1:0];
  logic live;  // out of reset for a cycle: advertisements wait for it, so no credit leaves while in reset
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) live <= 1'b0;
    else live <= 1'b1;
  logic [1:0] stg_owed;  // staging credits still to return: the advertisement
  logic stg_credit;
  assign stg_credit = bcast_release || (live && stg_owed != 0);
  assign staging.credit = stg_credit;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) stg_owed <= 2'd2;
    else stg_owed <= stg_owed + 2'(bcast_release) - 2'(stg_credit);

  assign in_room = !in_full[in_wr];
  assign start_accept = staging.put && in_room;
  logic west_wr_ok, north_wr_ok;
  assign west_wr_ok  = west_write_enable_i && in_room && !west_write_reset_i && ptr_A < GLOBAL_ELEMENTS;
  assign north_wr_ok = north_write_enable_i && in_room && !north_write_reset_i && ptr_B < GLOBAL_ELEMENTS;
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
      in_fresh <= '0;
      in_more <= '0;
      last_partial <= 1'b0;
      in_tile[0] <= '0;
      in_tile[1] <= '0;
      in_pack[0] <= '0;
      in_pack[1] <= '0;
      in_last <= '0;
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
        in_cached[in_wr] <= stg_cached;
        in_fresh[in_wr] <= !last_partial;
        in_more[in_wr] <= stg_partial;
        last_partial <= stg_partial;
        in_tile[in_wr] <= stg_tile;
        in_pack[in_wr] <= stg_pack;
        in_last[in_wr] <= stg_last;
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

  // ── L2: one slot per region; a put opens a fill, the fill's last set closes it, that set's broadcast returns the credit ──
  logic [1:0] wc_ret, wc_adv;  // wc_adv: the advertisement, sent once live
  for (genvar r = 0; r < 2; r++) begin : WC_LINK
    assign wc_ret[r] = (live && wc_adv[r]) || (bcast_release && in_cached[in_rd] && in_last[in_rd] && in_tile[in_rd][WCTW-1] == 1'(r));
    assign wc_region[r].credit = wc_ret[r];
  end
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) wc_adv <= 2'b11;
    else if (live) wc_adv <= 2'b00;
  function automatic logic [DATA_WIDTH-1:0] b_word(input int addr);
    return in_cached[in_rd] ? wcache[int'(in_tile[in_rd])*GLOBAL_ELEMENTS+addr] : mem_B[int'(in_rd)*GLOBAL_ELEMENTS+addr];
  endfunction
  assign north_queue_empty_o = (ptr_B == 0);

  // ── Broadcast: copy a full staging bank into every array's free operand bank, one tile row per cycle ──
  localparam int AK = COLLAPSE_K ? MATRIX_SIZE : TILE_SIZE;  // depth of each array's product
  localparam int ADD_LAT = sienna_fmt_pkg::add_lat(EXP_W, MAN_W);
  localparam int U = (AK < ADD_LAT + 1) ? AK : ADD_LAT + 1;  // partial sums per array pixel: the adder latency plus one
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
  logic ctrl_load_en, commit_q, commit_fresh_q, commit_more_q, set_launch, reduce_start;
  logic [2:0] commit_pack_q;
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
      commit_fresh_q <= 1'b0;
      commit_more_q <= 1'b0;
      commit_pack_q <= '0;
    end else begin
      commit_q <= bcast_release;  // lands with the last registered row write
      commit_fresh_q <= in_fresh[in_rd];
      commit_more_q <= in_more[in_rd];
      commit_pack_q <= in_pack[in_rd];
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

  // ── Result banks: FREE, WRITING from reduce start, FULL once written, FREE again after its last beat is pushed ──
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
  localparam int BEATS = GLOBAL_ELEMENTS / WIDE_READ;  // beats per pushed result
  logic [RESULT_BANKS-1:0] out_pk;  // per result bank: a packed set, pushed in the packed order
  logic [BIAS_Q-1:0] bias_pk;  // per bias queue entry: the sum is a packed set
  logic res_put, rq_last;  // a beat on the result link; the result's last
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
      if (res_put && rq_last) begin
        out_state[out_rd] <= R_FREE;
        out_rd <= next_bank(out_rd);
      end
    end
  end

  // ── Bias queue: sums reach the reducers in the order they were started ──
  logic bias_push;
  assign bias_push = start_accept && !last_partial;
  logic [ACC_W-1:0] bias_q[BIAS_Q][MATRIX_SIZE];
  logic [BIAS_Q-1:0] bias_qv;
  logic [$clog2(BIAS_Q)-1:0] bq_wr, bq_rd;
  logic [$clog2(BIAS_Q):0] bq_n;
  logic [MATRIX_SIZE-1:0][ACC_W-1:0] red_bias;  // bias of the set the reducers are starting
  always_comb begin
    for (int c = 0; c < MATRIX_SIZE; c++) red_bias[c] = bias_qv[bq_rd] ? bias_q[bq_rd][c] : '0;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      bq_wr   <= '0;
      bq_rd   <= '0;
      bq_n    <= '0;
      bias_qv <= '0;
      bias_pk <= '0;
    end else begin
      // One entry per sum: its first pass brings the bias, later passes of the same sum bring none.
      if (bias_push) begin
        for (int c = 0; c < MATRIX_SIZE; c++) bias_q[bq_wr][c] <= bias_i[c];
        bias_qv[bq_wr] <= stg_bias_v;
        bias_pk[bq_wr] <= stg_pack != 3'd0;
        bq_wr <= bq_wr + 1'b1;
      end
      if (reduce_start) bq_rd <= bq_rd + 1'b1;
      bq_n <= bq_n + (bias_push ? 1'b1 : 1'b0) - (reduce_start ? 1'b1 : 1'b0);
    end
  end
  always_ff @(posedge clk_i or negedge rstn_i)  // the pack flag goes with its sum from the bias queue to the result bank
    if (!rstn_i) out_pk <= '0;
    else if (reduce_start) out_pk[out_wr] <= bias_pk[bq_rd];

  logic mesh_busy;  // a set is somewhere between staging and a written result; for testbenches
  assign mesh_busy = (bstate != B_IDLE) || arrays_busy || reducers_busy;

  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] load_we_A, load_we_B;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][LW-1:0][DATA_WIDTH-1:0] load_data_A, load_data_B;

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      load_we_A <= '{default: 0};
      load_we_B <= '{default: 0};
    end else begin
      load_we_A <= '{default: 0};
      load_we_B <= '{default: 0};
      if (ctrl_load_en) begin
        automatic int sub_r = load_idx;  // tile row copied this cycle; block-local, so no register is inferred
        automatic int addr_calc, rr_L;
        if (COLLAPSE_K) begin
          // Tile row i takes row sub_r of its T x N slab of A; tile column j takes N/T rows of its N x T slab of B.
          for (int i_L = 0; i_L < TILES_PER_DIM; i_L++) begin
            for (int sub_c = 0; sub_c < MATRIX_SIZE; sub_c++) begin
              addr_calc = ((i_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + sub_c;
              load_data_A[i_L][0][sub_c] <= mem_A[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
            end
            load_we_A[i_L][0] <= 1;
          end
          for (int j_L = 0; j_L < TILES_PER_DIM; j_L++) begin
            for (int w_L = 0; w_L < MATRIX_SIZE; w_L++) begin
              rr_L = sub_r * TILES_PER_DIM + w_L / TILE_SIZE;
              addr_calc = rr_L * MATRIX_SIZE + j_L * TILE_SIZE + w_L % TILE_SIZE;
              load_data_B[0][j_L][w_L] <= b_word(addr_calc);
            end
            load_we_B[0][j_L] <= 1;
          end
        end else begin
          for (int i_L = 0; i_L < TILES_PER_DIM; i_L++) begin
            for (int k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
              for (int sub_c = 0; sub_c < TILE_SIZE; sub_c++) begin
                addr_calc = ((i_L * TILE_SIZE) + sub_r) * MATRIX_SIZE + ((k_L * TILE_SIZE) + sub_c);
                load_data_A[i_L][k_L][sub_c] <= mem_A[int'(in_rd)*GLOBAL_ELEMENTS+addr_calc];
              end
              load_we_A[i_L][k_L] <= 1;
            end
          end
          for (int k_L = 0; k_L < TILES_PER_DIM; k_L++) begin
            for (int j_L = 0; j_L < TILES_PER_DIM; j_L++) begin
              for (int sub_c = 0; sub_c < TILE_SIZE; sub_c++) begin
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
  logic [NUM_TILES-1:0][     OUT_W-1:0] sram_data_agg;
  logic [NUM_TILES-1:0][          31:0] sram_addr_bank;

  always_comb
    for (int p = 0; p < NUM_TILES; p++) sram_addr_bank[p] = sram_addr_agg[p];  // each reducer adds its own bank offset

  // ── L3: the oldest result goes out as BEATS wide beats, one read a cycle while a credit is held past this cycle's put ──
  // Beat i word k is element k * BEATS + i; packed, it is column k % N of row (k / N) * BEATS + i, so a lane holds a column block.
  localparam int RES_W = WIDE_READ * OUT_W + 3;
  logic [$clog2(BEATS + 1)-1:0] pb_idx;  // next beat of the oldest result to read
  logic res_rd, rq_first, rq_pk;
  logic [$clog2(RES_MAX + 1)-1:0] res_cnt;  // result credits held
  logic [WIDE_READ-1:0][OUT_W-1:0] res_words;
  credit_counter #(.MAX(RES_MAX), .CRW(RES_CRW)) res_cc (.clk_i(clk_i), .rstn_i(rstn_i), .put_i(res_put), .credit_i(result.credit),
                                                        .has_credit_o(), .count_o(res_cnt));
  assign res_rd = out_full[out_rd] && (int'(pb_idx) < BEATS) && (int'(res_cnt) > int'(res_put));
  assign result.put  = res_put;
  assign result.data = {rq_pk, rq_last, rq_first, res_words};
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) begin
      pb_idx   <= '0;
      rq_first <= 1'b0;
      rq_last  <= 1'b0;
      rq_pk    <= 1'b0;
    end else begin
      if (res_rd) begin
        rq_first <= pb_idx == 0;
        rq_last  <= int'(pb_idx) == BEATS - 1;
        rq_pk    <= out_pk[out_rd];
        pb_idx   <= pb_idx + 1'b1;
      end
      if (res_put && rq_last) pb_idx <= '0;  // reads stopped at the last beat, so this never meets a read
    end

  logic [WIDE_READ-1:0][31:0] wide_addr;
  always_comb
    for (int k = 0; k < WIDE_READ; k++)
      wide_addr[k] = int'(out_rd) * GLOBAL_ELEMENTS + (out_pk[out_rd]
                     ? ((k / MATRIX_SIZE) * BEATS + int'(pb_idx)) * MATRIX_SIZE + k % MATRIX_SIZE
                     : k * BEATS + int'(pb_idx));

`ifndef SYNTHESIS  // parameter checks; synthesis tools ignore or reject initial blocks
  initial
    if (GLOBAL_ELEMENTS % WIDE_READ != 0)
      $error("SystolicMesh: WIDE_READ (%0d) must divide N*N (%0d)", WIDE_READ, GLOBAL_ELEMENTS);
  // Interface widths are not elaboration constants in Verilator, so the link widths are checked at time 0.
  initial
    if ($bits(staging.data) != STG_W || $bits(result.data) != RES_W || $bits(result.credit) != RES_CRW)
      $fatal(1, "SystolicMesh: links need staging.data %0d bits, result.data %0d, result.credit RES_CRW %0d", STG_W, RES_W, RES_CRW);
`endif
  if (RES_CRW > $clog2(RES_MAX + 1)) begin : G_BAD_RES_CRW  // credit_counter would drop the credit's high bits
    $fatal(1, "SystolicMesh: RES_CRW %0d is wider than the result counter's %0d bits (RES_MAX %0d)", RES_CRW, $clog2(RES_MAX + 1), RES_MAX);
  end

  MeshOutputSram #(
      .DEPTH(RESULT_BANKS * GLOBAL_ELEMENTS),
      .DATA_WIDTH(OUT_W),
      .NUM_PORTS(NUM_TILES),
      .WIDE(WIDE_READ)
  ) output_mem (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .we_i(sram_we_agg),
      .waddr_i(sram_addr_bank),
      .wdata_i(sram_data_agg),
      .wide_enable_i(res_rd),
      .wide_addr_i(wide_addr),
      .wide_data_o(res_words),
      .wide_valid_o(res_put)
  );

  // Per output tile: U partials from each depth slice, flattened for its reducer.
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][RPU-1:0][ACC_W-1:0] t_data;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0] t_ren;
  logic [TILES_PER_DIM-1:0][TILES_PER_DIM-1:0][ACC_W-1:0] t_bias;  // the bias of the pixel being summed
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
            .EXP_W(EXP_W),
            .MAN_W(MAN_W),
            .ACC_W(ACC_W),
            .OUT_W(OUT_W),
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
            logic [U-1:0][ACC_W-1:0] rd;
            SystolicArray #(
                .N(TILE_SIZE),
                .K(AK),
                .EXP_W(EXP_W),
                .MAN_W(MAN_W),
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_W(ACC_W),
                .WEST_WORDS(LW),
                .NORTH_WORDS(LW),
                .U(U),
                .BANKS(ACC_BANKS),
                .COL0(COLLAPSE_K ? j * TILE_SIZE : 0)
            ) tile (
                .clk_i(clk_i),
                .rstn_i(rstn_i),
                .west_write_enable_i(load_we_A[i][k]),
                .west_write_data_i(load_data_A[i][k]),
                .north_write_enable_i(load_we_B[k][j]),
                .north_write_data_i(load_data_B[k][j]),
                .commit_i(commit_q),
                .commit_fresh_i(commit_fresh_q),
                .commit_more_i(commit_more_q),
                .commit_pack_i(commit_pack_q),
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
  // The accept's terms, registered: sampled assertion values miss a combinational start_accept when the host drives the start at the edge.
  logic sa_q, sa_partial_q, sa_last_partial_q, sa_bias_v_q;  // sa_last_partial_q: last_partial before this accept updated it
  // Per region: a fill was put and its last set not yet; the put only feeds these checks, the credit needs only the last set.
  logic [1:0] wc_put, wc_open;
  for (genvar r = 0; r < 2; r++) begin : WC_PUT
    assign wc_put[r] = wc_region[r].put;
  end
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) wc_open <= 2'b00;
    else
      for (int r = 0; r < 2; r++)
        if (wc_put[r]) wc_open[r] <= 1'b1;
        else if (start_accept && stg_cached && stg_last && stg_tile[WCTW-1] == 1'(r)) wc_open[r] <= 1'b0;
  logic sa_cached_q, sa_open_q, sa_last_q;  // a cached set, its region's fill was open, it is marked the fill's last
  logic [2:0] sa_shift_q;
  logic [$clog2(BIAS_Q):0] sa_bq_n_q;
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {sa_q, sa_partial_q, sa_last_partial_q, sa_bias_v_q, sa_shift_q, sa_bq_n_q, sa_cached_q, sa_open_q, sa_last_q} <= '0;
    else {sa_q, sa_partial_q, sa_last_partial_q, sa_bias_v_q, sa_shift_q, sa_bq_n_q, sa_cached_q, sa_open_q, sa_last_q} <=
             {start_accept, stg_partial, last_partial, stg_bias_v, stg_pack, bq_n, stg_cached, wc_open[stg_tile[WCTW-1]], stg_last};
  // A staging put, a row write and a cache write, each with its terms, registered for the same reason: a host drives them at the edge too.
  logic sp_q, sp_full_q, rw_q, rw_full_q, wcw_q, wcw_open_q, wcw_hit_q;
  logic wc_hit;  // the cache write's tile is one a staged set reads
  always_comb begin
    wc_hit = 1'b0;
    for (int b = 0; b < 2; b++)
      if (in_full[b] && in_cached[b] && int'(in_tile[b]) == int'(wc_write_addr_i) / GLOBAL_ELEMENTS) wc_hit = 1'b1;
  end
  always_ff @(posedge clk_i or negedge rstn_i)
    if (!rstn_i) {sp_q, sp_full_q, rw_q, rw_full_q, wcw_q, wcw_open_q, wcw_hit_q} <= '0;
    else {sp_q, sp_full_q, rw_q, rw_full_q, wcw_q, wcw_open_q, wcw_hit_q} <=
             {staging.put, in_full[in_wr], west_write_enable_i || north_write_enable_i, in_full[in_wr], wc_write_enable_i,
              wc_open[wc_write_addr_i[WCAW-1]], wc_hit};
  // Handshake invariants; live only with --assert.
  a_result_bank_free: assert property (@(posedge clk_i) disable iff (!rstn_i) reduce_start |-> out_state[out_wr] == R_FREE)
    else $error("SystolicMesh: a reduce started into a result bank that is not free");
  a_written_in_order: assert property (@(posedge clk_i) disable iff (!rstn_i) set_done |-> out_state[wr_bank_done] == R_WRITING)
    else $error("SystolicMesh: a set finished writing into a bank that was not being written");
  a_bias_first_pass: assert property (@(posedge clk_i) disable iff (!rstn_i) (sa_q && sa_last_partial_q) |-> !sa_bias_v_q)
    else $error("SystolicMesh: a bias came with a later pass of a sum; it belongs with the first");
  a_bias_queue_room: assert property (@(posedge clk_i) disable iff (!rstn_i) (sa_q && !sa_last_partial_q) |-> sa_bq_n_q < BIAS_Q)
    else $error("SystolicMesh: bias queue overflow");
  a_bias_queue_held: assert property (@(posedge clk_i) disable iff (!rstn_i) reduce_start |-> bq_n != 0)
    else $error("SystolicMesh: a reduce started with no bias queued");
  a_wc_bus: assert property (@(posedge clk_i) disable iff (!rstn_i) !(wc_write_enable_i && north_write_enable_i))
    else $error("SystolicMesh: a cache write and a B write share the north bus in one cycle");
  a_wc_fill_open: assert property (@(posedge clk_i) disable iff (!rstn_i) wcw_q |-> wcw_open_q)
    else $error("SystolicMesh: cache write into a region with no open fill (no put on its link, or its last set already put)");
  a_wc_tile_free: assert property (@(posedge clk_i) disable iff (!rstn_i) wcw_q |-> !wcw_hit_q)
    else $error("SystolicMesh: cache write into a tile a staged set still reads");
  a_wc_last_cached: assert property (@(posedge clk_i) disable iff (!rstn_i) (sa_q && sa_last_q) |-> sa_cached_q)
    else $error("SystolicMesh: wc_last on an uncached set: no region is released and its credit never returns");
  a_wc_set_open: assert property (@(posedge clk_i) disable iff (!rstn_i) (sa_q && sa_cached_q) |-> sa_open_q)
    else $error("SystolicMesh: a cached set reads a region with no open fill");
  a_stage_room: assert property (@(posedge clk_i) disable iff (!rstn_i) sp_q |-> !sp_full_q)
    else $error("SystolicMesh: staging put with both banks full, a put without a credit; ignored");
  a_row_room: assert property (@(posedge clk_i) disable iff (!rstn_i) rw_q |-> !rw_full_q)
    else $error("SystolicMesh: row write with no free staging bank, rows without a credit; dropped");
  a_staging_bank_full: assert property (@(posedge clk_i) disable iff (!rstn_i) bcast_release |-> in_full[in_rd])
    else $error("SystolicMesh: BROADCAST copied an empty staging bank");
  a_arrays_ready_on_launch: assert property (@(posedge clk_i) disable iff (!rstn_i) (load_we_A != '0) |-> arrays_load_ready)
    else $error("SystolicMesh: broadcast wrote an array whose operand bank was not free");
  a_pack_range: assert property (@(posedge clk_i) disable iff (!rstn_i) sa_q |-> int'(sa_shift_q) < $clog2(MATRIX_SIZE))
    else $error("SystolicMesh: pack shift %0d leaves blocks narrower than 2 of N=%0d", sa_shift_q, MATRIX_SIZE);
  a_pack_collapsed: assert property (@(posedge clk_i) disable iff (!rstn_i) (sa_q && sa_shift_q != 0) |-> COLLAPSE_K != 0)
    else $error("SystolicMesh: a packed set on the collapse-k 0 mesh");
  a_pack_one_pass: assert property (@(posedge clk_i) disable iff (!rstn_i)
                                    (sa_q && sa_shift_q != 0) |-> (!sa_partial_q && !sa_last_partial_q))
    else $error("SystolicMesh: a packed set is part of an accumulated sum");
  a_wide_packed: assert property (@(posedge clk_i) disable iff (!rstn_i) (res_rd && out_pk[out_rd]) |-> (WIDE_READ % MATRIX_SIZE == 0))
    else $error("SystolicMesh: a packed result's push needs N (%0d) to divide WIDE_READ (%0d)", MATRIX_SIZE, WIDE_READ);
`ifdef ASSERT_SELFTEST
  a_selftest: assert property (@(posedge clk_i) disable iff (!rstn_i) 1'b0)
    else $error("SystolicMesh: assertion self-test fired, so assertions are live");
`endif
`endif

endmodule
