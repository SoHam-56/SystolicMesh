`timescale 1ns / 100ps

// FAULT 1: rows and a staging put with no credit; 2: a cache write after its region's last set, then a cached set on an unfilled region; 3: a result link one bit narrow.
module TB_SystolicMesh #(
    parameter int FAULT = 0,
    parameter int WR    = 0,  // words per result beat; 0 is N
    parameter int RC    = 0   // the most result credits +res_slots may advertise; 0 is one set's beats
);

  localparam int EXP_W = 8;  // patched by regression.py --format
  localparam int MAN_W = 23;
  localparam int COLLAPSE_K = 1;  // patched by regression.py --collapse-k
  localparam DATA_WIDTH = 1 + EXP_W + MAN_W;
  localparam int ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W);  // result words: int32 in int8, DATA_WIDTH in floats
  localparam CLK_PERIOD = 10;

  localparam MATRIX_SIZE = 16;
  localparam TILE_SIZE = 4;
  localparam SRAM_SIZE = MATRIX_SIZE * MATRIX_SIZE;
  localparam int HOST_WORDS = MATRIX_SIZE;  // words per host write: one matrix row

  localparam int NUM_TEST_SETS = 5;

  // Reset once at time zero only. Resetting per set hides every re-arm defect.
  localparam bit B2B_MODE = 1'b1;

  // A matmul is ~120 cycles at N=16; 500k made every hung set a 20-minute wait.
  localparam int TIMEOUT_CYCLES = 20_000;

  // Every format compares bit for bit: the expected results come from mesh_model, the hardware's own arithmetic.

  // ── Links: staging sets (L1), weight-cache regions (L2), result beats (L3) ──
  localparam int WIDE_READ = (WR == 0) ? MATRIX_SIZE : WR;
  localparam int BEATS = SRAM_SIZE / WIDE_READ;  // beats per result
  localparam int RES_CAP = (RC == 0) ? BEATS : RC;
  localparam int RES_CRW = $clog2(RES_CAP + 1);  // the consumer may free every slot in one cycle
  localparam int WC_TILES = 128;
  localparam int WCTW = $clog2(WC_TILES);
  localparam int WCAW = $clog2(WC_TILES * SRAM_SIZE);
  localparam int HALF = WC_TILES / 2;  // first tile of cache region 1
  localparam int STG_W = WCTW + 7;  // {wc_last, weight_tile, weight_cached, pack_shift[2:0], bias_valid, partial}
  localparam int RES_W = WIDE_READ * ACC_W + 3;  // {packed, last, first, WIDE_READ words}
  localparam int MAX_RES = 64;  // results one run collects

  reg clk, rstn;

  reg n_we, w_we, n_rst, w_rst;
  reg [HOST_WORDS-1:0][DATA_WIDTH-1:0] n_data, w_data;
  wire n_empty, w_empty, complete;
  reg wc_we = 1'b0;  // cache write of n_data at wc_addr
  reg [WCAW-1:0] wc_addr = '0;

  reg     [     ACC_W-1:0] expected_mem          [0:SRAM_SIZE-1];
  reg                              bias_v = 1'b0;  // the set's bias (int8: matrixBias<suffix>.mem), taken by the mesh with the put
  reg [MATRIX_SIZE-1:0][ACC_W-1:0] bias_d = '0;
  reg [2:0] pack_d = '0;  // the set's pack shift (packShift<suffix>.mem), taken with the put

  credit_link_if #(.DATA_W(STG_W), .CRW(1)) stg ();
  credit_link_if #(.DATA_W(1), .CRW(1)) wcl[2] ();
  credit_link_if #(.DATA_W(RES_W - ((FAULT == 3) ? 1 : 0)), .CRW(RES_CRW)) rsl ();

  // ── The host: producer of L1 and L2, a credit counter on each ─────────────
  logic stg_has;
  logic [1:0] stg_cnt;
  logic [1:0] wc_has;
  logic drained = 1'b0;  // every link idle with its credits back; the checkers then test it
  credit_counter #(.MAX(2), .CRW(1)) stg_cc (.clk_i(clk), .rstn_i(rstn), .put_i(stg.put), .credit_i(stg.credit), .has_credit_o(stg_has),
                                            .count_o(stg_cnt));
  credit_link_checker #(.SLOTS(2)) chk_stg (.clk_i(clk), .rstn_i(rstn), .drained_i(drained), .lnk(stg));
  for (genvar r = 0; r < 2; r++) begin : WCP
    assign wcl[r].data = 1'b0;
    credit_counter #(.MAX(1), .CRW(1)) cc (.clk_i(clk), .rstn_i(rstn), .put_i(wcl[r].put), .credit_i(wcl[r].credit),
                                          .has_credit_o(wc_has[r]), .count_o());
    credit_link_checker #(.SLOTS(1)) chk (.clk_i(clk), .rstn_i(rstn), .drained_i(drained), .lnk(wcl[r]));
  end
  int res_slots, stall_pct;  // the consumer's advertisement and its stall rate, from +res_slots and +stall_pct
  credit_link_checker #(.SLOTS(RES_CAP)) chk_res (.clk_i(clk), .rstn_i(rstn), .drained_i(drained && res_slots == RES_CAP), .lnk(rsl));

  // ── Verification counters ──────────────────────────────────────────────────
  int                      total_sets_run = 0;
  int                      sets_passed = 0;
  int                      sets_failed = 0;
  int                      total_elements = 0;

  // ── Cycle-count tracking ───────────────────────────────────────────────────
  // Hardware cycle counter: counts from the staging put until complete asserts.
  // Declared as longint to handle large cycle counts without overflow.
  longint                  cycle_count;
  logic                    counting;

  // Per-set cycle log (max NUM_TEST_SETS entries)
  longint                  set_cycles            [        0:255];

  // complete is a level held through DONE, so it is still high from the previous set
  // when start is pulsed. Stopping on the level made every set after the first
  // measure 0 cycles; stop on its rising edge instead.
  logic complete_d;
  always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      cycle_count <= 0;
      counting    <= 0;
      complete_d  <= 0;
    end else begin
      complete_d <= complete;
      if (stg.put) begin  // latch the put — begin counting next cycle
        cycle_count <= 0;
        counting    <= 1;
      end else if (counting && complete && !complete_d) begin  // stop on completion edge
        counting <= 0;
      end else if (counting) begin
        cycle_count <= cycle_count + 1;
      end
    end
  end

  // ─────────────────────────────────────────────────────────────────────────
  SystolicMesh #(
      .MATRIX_SIZE(MATRIX_SIZE),
      .TILE_SIZE  (TILE_SIZE),
      .EXP_W      (EXP_W),
      .MAN_W      (MAN_W),
      .COLLAPSE_K (COLLAPSE_K),
      .DATA_WIDTH (DATA_WIDTH),
      .ACC_W      (ACC_W),
      .WIDE_READ  (WIDE_READ),
      .HOST_WORDS (HOST_WORDS),
      .WC_TILES   (WC_TILES),
      .RES_MAX    (RES_CAP),
      .RES_CRW    (RES_CRW)
  ) dut (
      .clk_i(clk),
      .rstn_i(rstn),
      .staging(stg),
      .bias_i(bias_d),
      .wc_region(wcl),
      .wc_write_enable_i(wc_we),
      .wc_write_addr_i(wc_addr),

      .north_write_enable_i(n_we),
      .north_write_data_i  (n_data),
      .north_write_reset_i (n_rst),

      .west_write_enable_i(w_we),
      .west_write_data_i  (w_data),
      .west_write_reset_i (w_rst),

      .north_queue_empty_o(n_empty),
      .west_queue_empty_o(w_empty),
      .matrix_mult_complete_o(complete),
      .collection_active_o(),
      .result(rsl)
  );

  // ── Clock ─────────────────────────────────────────────────────────────────
  initial begin
    clk = 0;
    forever #(CLK_PERIOD / 2) clk = ~clk;
  end

  // ── The consumer of L3: advertises +res_slots beats, takes them at a random rate (+stall_pct), frees slots in batches ──
  // Runs on the falling edge: it samples the mesh's registered put and drives credit between edges.
  bit res_hold = 0;  // the consumer takes nothing
  logic [RES_W-1:0] res_fifo[$];
  int res_owed = 0;  // slots freed and not yet credited back
  int res_got = 0, res_beat = 0, res_bad = 0;  // results collected, beats of the next one, framing errors
  logic [ACC_W-1:0] res_store[MAX_RES][SRAM_SIZE];
  bit res_pk[MAX_RES];
  bit res_hit[SRAM_SIZE];  // elements of the result being collected

  task automatic take_beat(input logic [RES_W-1:0] d);
    bit pk, last, first;
    int e;
    {pk, last, first} = d[RES_W-1-:3];
    if (first != (res_beat == 0) || last != (res_beat == BEATS - 1)) begin
      res_bad++;
      $display("  [FAIL] Result %0d beat %0d: first=%0b last=%0b", res_got, res_beat, first, last);
    end
    if (res_beat == 0) res_hit = '{default: 0};
    for (int k = 0; k < WIDE_READ; k++) begin
      e = pk ? ((k / MATRIX_SIZE) * BEATS + res_beat) * MATRIX_SIZE + k % MATRIX_SIZE : k * BEATS + res_beat;  // the mesh's push order
      if (e >= SRAM_SIZE || res_hit[e]) begin
        res_bad++;
        $display("  [FAIL] Result %0d beat %0d word %0d: element %0d out of range or pushed twice", res_got, res_beat, k, e);
      end else begin
        res_hit[e] = 1'b1;
        if (res_got < MAX_RES) res_store[res_got][e] = d[k*ACC_W+:ACC_W];
      end
    end
    res_owed++;
    if (res_beat == BEATS - 1) begin
      if (res_got < MAX_RES) res_pk[res_got] = pk;
      res_got++;
      res_beat = 0;
    end else res_beat++;
  endtask

  initial begin
    rsl.credit = '0;
    forever begin
      @(negedge clk);
      if (!rstn) begin
        rsl.credit = '0;
        res_fifo.delete();
        res_owed = res_slots;  // the advertisement, sent once reset ends
        res_beat = 0;
      end else begin
        // Pop before taking this cycle's beat: a beat is freed a cycle after it arrives at the earliest, as in hardware.
        if (!res_hold && res_fifo.size() != 0 && $urandom_range(99) >= stall_pct) take_beat(res_fifo.pop_front());
        if (rsl.put) res_fifo.push_back(RES_W'(rsl.data));
        if (res_fifo.size() > res_slots) begin
          res_bad++;
          $display("  [FAIL] Result link: %0d beats held with %0d slots advertised", res_fifo.size(), res_slots);
        end
        if (res_owed != 0 && (res_owed >= 3 || res_fifo.size() == 0)) begin  // several slots freed in one cycle
          rsl.credit = RES_CRW'(res_owed);
          res_owed = 0;
        end else rsl.credit = '0;
      end
    end
  end

  // ── Reset ─────────────────────────────────────────────────────────────────
  task apply_reset();
    begin
      rstn       = 0;
      n_we       = 0;
      n_rst      = 0;
      n_data     = 0;
      w_we       = 0;
      w_rst      = 0;
      w_data     = 0;
      wc_we      = 0;
      stg.put    = 0;
      stg.data   = '0;
      wcl[0].put = 0;
      wcl[1].put = 0;
      repeat (5) @(posedge clk);
      rstn = 1;
      repeat (5) @(posedge clk);
    end
  endtask

  // ── Queue loaders ─────────────────────────────────────────────────────────
  task load_west_queue(input string filename);
    integer fh, res, cnt;
    reg [DATA_WIDTH-1:0] tmp;
    begin
      fh = $fopen(filename, "r");
      if (!fh) begin
        $display("  [Error] Could not open WEST file: %s", filename);
        $finish;
      end
      w_rst = !B2B_MODE;  // back-to-back: the mesh must rewind its own pointer
      @(posedge clk);
      w_rst = 0;
      @(posedge clk);
      cnt = 0;
      while (!$feof(
          fh
      )) begin
        res = $fscanf(fh, "%h", tmp);
        if (res == 1) begin
          w_data[cnt % HOST_WORDS] = tmp;
          cnt++;
          if (cnt % HOST_WORDS == 0) begin
            w_we = 1;
            @(posedge clk);
            w_data = '0;
          end
        end
      end
      if (cnt % HOST_WORDS != 0) begin  // a short last write; the rest of the bank is zero anyway
        w_we = 1;
        @(posedge clk);
      end
      w_we = 0;
      @(posedge clk);
      $fclose(fh);
    end
  endtask

  task load_north_queue(input string filename);
    integer fh, res, cnt;
    reg [DATA_WIDTH-1:0] tmp;
    begin
      fh = $fopen(filename, "r");
      if (!fh) begin
        $display("  [Error] Could not open NORTH file: %s", filename);
        $finish;
      end
      n_rst = !B2B_MODE;  // back-to-back: the mesh must rewind its own pointer
      @(posedge clk);
      n_rst = 0;
      @(posedge clk);
      cnt = 0;
      while (!$feof(
          fh
      )) begin
        res = $fscanf(fh, "%h", tmp);
        if (res == 1) begin
          n_data[cnt % HOST_WORDS] = tmp;
          cnt++;
          if (cnt % HOST_WORDS == 0) begin
            n_we = 1;
            @(posedge clk);
            n_data = '0;
          end
        end
      end
      if (cnt % HOST_WORDS != 0) begin  // a short last write; the rest of the bank is zero anyway
        n_we = 1;
        @(posedge clk);
      end
      n_we = 0;
      @(posedge clk);
      $fclose(fh);
    end
  endtask

  // ── Result verification: the next collected result against a golden file ─
  int res_next = 0;  // next collected result to check
  bit put_pk[MAX_RES];  // per put, in order: the set is packed
  int n_puts = 0;

  task automatic wait_result();
    for (int w = 0; w < TIMEOUT_CYCLES && res_got <= res_next; w++) @(posedge clk);
    if (res_got <= res_next) begin
      $display("  [FATAL] Timeout waiting for result %0d", res_next);
      $finish;
    end
  endtask

  task verify_results(input string filename, output int err_count);
    integer fh, i, res;
    reg [ACC_W-1:0] exp_val, actual_val;
    begin
      $display("  [Verify] Checking against %s...", filename);
      fh = $fopen(filename, "r");
      if (!fh) begin
        $display("  [Error] Could not open EXPECTED file: %s", filename);
        $finish;
      end
      i = 0;
      while (!$feof(
          fh
      ) && i < SRAM_SIZE) begin
        res = $fscanf(fh, "%h", exp_val);
        if (res == 1) begin
          expected_mem[i] = exp_val;
          i++;
        end
      end
      $fclose(fh);

      wait_result();
      err_count = 0;
      for (i = 0; i < SRAM_SIZE; i++) begin
        actual_val = res_store[res_next][i];
        total_elements++;

        if (actual_val !== expected_mem[i]) begin
          $display("    [FAIL]     Addr %0d: Exp=0x%h, Act=0x%h", i, expected_mem[i], actual_val);
          err_count++;
        end
      end
      if (res_pk[res_next] != put_pk[res_next]) begin
        $display("    [FAIL]     Result %0d pushed with packed=%0b, the set was put with packed=%0b", res_next, res_pk[res_next],
                 put_pk[res_next]);
        err_count++;
      end
      res_next++;

      if (err_count == 0) $display("  [Result] Set Passed.");
      else $display("  [Result] Set FAILED with %0d mismatches.", err_count);
    end
  endtask

  // ── Per-set stimulus file names ───────────────────────────────────────────
  task automatic set_files(input int s, output string f_a, output string f_b, output string f_c);
    if (NUM_TEST_SETS == 1) begin
      f_a = "matrixA.mem";
      f_b = "matrixB.mem";
      f_c = "matrixC.mem";
    end else begin
      f_a = $sformatf("matrixA_%0d.mem", s);
      f_b = $sformatf("matrixB_%0d.mem", s);
      f_c = $sformatf("matrixC_%0d.mem", s);
    end
  endtask

  // ── Per-set bias: the mesh samples bias_i with the put; matrixBias<suffix>.mem exists only in int8 ────────
  task automatic drive_bias(input int s);
    string f;
    integer fh, res;
    reg [ACC_W-1:0] tmp;
    f = (NUM_TEST_SETS == 1) ? "matrixBias.mem" : $sformatf("matrixBias_%0d.mem", s);
    bias_d = '0;
    bias_v = 1'b0;
    fh = $fopen(f, "r");
    if (fh) begin
      for (int c = 0; c < MATRIX_SIZE; c++) begin
        res = $fscanf(fh, "%h", tmp);
        if (res != 1) begin
          $display("  [Error] %s: fewer than %0d bias words", f, MATRIX_SIZE);
          $finish;
        end
        bias_d[c] = tmp;
      end
      bias_v = 1'b1;
      $fclose(fh);
    end
  endtask

  // ── Per-set pack shift: packShift<suffix>.mem exists only for packed sets ─────────────────────────────────
  task automatic drive_pack(input int s);
    string f;
    integer fh, res;
    reg [31:0] tmp;
    f = (NUM_TEST_SETS == 1) ? "packShift.mem" : $sformatf("packShift_%0d.mem", s);
    pack_d = '0;
    fh = $fopen(f, "r");
    if (fh) begin
      res = $fscanf(fh, "%h", tmp);
      pack_d = tmp[2:0];
      $fclose(fh);
    end
  endtask

  // ── Link puts: on the falling edge, so the mesh, the counters and the checkers all sample them alike ─────
  task automatic wait_stg_credit();  // the credit reserves the staging bank the rows go into
    @(negedge clk);
    while (!stg_has) @(negedge clk);
  endtask

  task automatic stg_put(input bit cached, input int tile, input bit last);
    @(negedge clk);
    stg.data = {last, WCTW'(tile), cached, pack_d, bias_v, 1'b0};
    stg.put  = 1'b1;
    if (n_puts < MAX_RES) put_pk[n_puts] = (pack_d != 0);
    n_puts++;
    @(negedge clk);
    stg.put = 1'b0;
  endtask

  task automatic wc_put(input int r);
    @(negedge clk);
    if (r == 0) wcl[0].put = 1'b1;
    else wcl[1].put = 1'b1;
    @(negedge clk);
    wcl[0].put = 1'b0;
    wcl[1].put = 1'b0;
  endtask

  // Waits for a staging credit, writes set s's rows (A only when cached), then puts it.
  task automatic stage_set(input int s, input bit cached, input int tile, input bit last);
    string f_a, f_b, f_c;
    set_files(s, f_a, f_b, f_c);
    wait_stg_credit();
    if (cached) load_west_queue(f_a);
    else
      fork
        load_west_queue(f_a);
        load_north_queue(f_b);
      join
    drive_bias(s);
    drive_pack(s);
    stg_put(cached, tile, last);
  endtask

  // Every link idle: the host holds both staging and both region credits, the mesh every result credit.
  task automatic drain_check(input string pass);
    repeat (TILE_SIZE + 20) @(negedge clk);
    if (stg_cnt != 2 || wc_has != 2'b11 || int'(dut.res_cnt) != res_slots || res_fifo.size() != 0)
      $display("  [FAIL] %s: links not drained: staging credits %0d of 2, region credits %b, result credits %0d of %0d", pass, stg_cnt,
               wc_has, dut.res_cnt, res_slots);
    drained = 1'b1;
    repeat (2) @(negedge clk);
    drained = 1'b0;
  endtask

  // ── Single test set ───────────────────────────────────────────────────────
  task execute_test_set(input int set_id);
    string f_a, f_b, f_c;
    int set_errors;
    bit load_empty;
    longint cycles_taken;
    begin
      set_files(set_id, f_a, f_b, f_c);

      $display("\n=========================================");
      $display("STARTING TEST SET %0d", set_id);
      $display("=========================================");
      $display("  Inputs: %s, %s", f_a, f_b);

      if (!B2B_MODE || set_id == 0) apply_reset();

      wait_stg_credit();
      fork
        load_west_queue(f_a);
        load_north_queue(f_b);
      join

      repeat (10) @(posedge clk);

      // sienna_top refuses to start on an empty queue, so a loaded queue reading empty is a failure.
      load_empty = w_empty || n_empty;
      if (load_empty) $display("  [FAIL] Queue reads empty after load (west=%0b north=%0b)", w_empty, n_empty);

      $display("  [Action] Starting Matrix Mult...");
      drive_bias(set_id);
      drive_pack(set_id);
      stg_put(1'b0, 0, 1'b0);

      // Timeout protection
      fork
        begin
          // complete is a level that stays high while the mesh sits in DONE, so it
          // is still asserted from the previous set when start is pulsed. Wait for
          // the mesh to leave DONE first, otherwise this returns immediately and the
          // results are read before the new matmul has run.
          wait (!complete);
          wait (complete);
        end
        begin
          repeat (TIMEOUT_CYCLES) @(posedge clk);
          if (!complete) begin
            $display("  [FATAL] Timeout waiting for completion signal!");
            $finish;
          end
        end
      join_any
      disable fork;

      // Capture cycle count at the clock edge after complete asserts
      @(posedge clk);
      cycles_taken       = cycle_count;
      set_cycles[set_id] = cycles_taken;

      $display("  [Perf]   Cycles to complete : %0d", cycles_taken);
      $display("  [Perf]   Wall time @ %0dns clk: %0d ns", CLK_PERIOD, cycles_taken * CLK_PERIOD);

      $display("  [Action] Processing Complete. Verifying...");
      verify_results(f_c, set_errors);
      if (load_empty) set_errors++;

      total_sets_run++;
      if (set_errors == 0) sets_passed++;
      else sets_failed++;

      repeat (20) @(posedge clk);
    end
  endtask

  // ── Streaming: host, mesh and consumer run concurrently ───────────────────
  // Sampled on the falling edge from registered state only, never a combinational view of start.
  int  in_overlap = 0, out_overlap = 0, n_launched = 0, n_completed = 0;
  bit  streaming = 0, count_bad = 0;
  wire mesh_busy = dut.mesh_busy;  // a set between staging and a written result
  initial forever begin
    @(negedge clk);
    if (streaming) begin
      if ((w_we || n_we) && mesh_busy) in_overlap++;
      if (rsl.put && mesh_busy) out_overlap++;
      if (dut.set_launch) n_launched++;  // one cycle per set, as the broadcast starts
      if (dut.set_done) n_completed++;
      if (n_completed > n_launched) count_bad = 1;
    end
  end

  task automatic stream_all_sets();
    longint t0;
    $display("\n[STAGE] STREAMING: %0d sets, host / mesh / consumer concurrent", NUM_TEST_SETS);
    streaming = 1;
    t0 = $time;
    fork
      begin
        fork
          begin : producer
            for (int s = 0; s < NUM_TEST_SETS; s++) stage_set(s, 1'b0, 0, 1'b0);
          end
          begin : consumer
            string f_a, f_b, f_c;
            int errs;
            for (int s = 0; s < NUM_TEST_SETS; s++) begin
              set_files(s, f_a, f_b, f_c);
              wait_result();
              $display("  [Stream] set %0d collected @%0t", s, $time);
              verify_results(f_c, errs);
              total_sets_run++;
              if (errs == 0) sets_passed++;
              else sets_failed++;
            end
          end
        join
      end
      begin : watchdog
        repeat (TIMEOUT_CYCLES * NUM_TEST_SETS) @(posedge clk);
        $display("  [FATAL] Timeout in the streaming pass");
        $finish;
      end
    join_any
    disable fork;
    streaming = 0;
    $display("  [Stream] %0d sets in %0d cycles", NUM_TEST_SETS, ($time - t0) / CLK_PERIOD);
    $display("  [Stream] host loading while mesh busy: %0d cycles", in_overlap);
    $display("  [Stream] results pushed while mesh busy: %0d cycles", out_overlap);
    if (in_overlap == 0) $display("  [FAIL] Overlap: host never loaded a set while the mesh was busy");
    if (out_overlap == 0) $display("  [FAIL] Overlap: the mesh never pushed a result while busy");
    if (count_bad || n_completed != NUM_TEST_SETS || n_launched != NUM_TEST_SETS)
      $display("  [FAIL] %0d sets launched and %0d completed, expected %0d each", n_launched,
               n_completed, NUM_TEST_SETS);
    drain_check("stream");
  endtask

  // ── The producer waits: the consumer holds, so no staging credit returns once every bank is full ─────
  // Sets the mesh holds with no result taken: 2 staging banks, 2 operand and ACC_BANKS partial-sum banks per array, RESULT_BANKS.
  localparam int MESH_SETS = 2 + 2 + 4 + 4;
  localparam int WAIT_CYCLES = TIMEOUT_CYCLES / 4;  // no credit for this long: the mesh is full

  task automatic producer_waits_test();
    string f_a, f_b, f_c;
    int errs, q, held, waited;
    held = MESH_SETS + res_slots / BEATS;  // whole results in the consumer's slots free their banks
    $display("\n[STAGE] PRODUCER WAITS: the consumer holds; the host stages sets until no staging credit returns");
    n_launched  = 0;
    n_completed = 0;
    count_bad   = 0;
    streaming   = 1;
    res_hold    = 1;
    fork
      begin
        q = 0;
        forever begin
          waited = 0;
          @(negedge clk);
          while (!stg_has && waited < WAIT_CYCLES) begin
            @(negedge clk);
            waited++;
          end
          if (!stg_has) break;
          stage_set(q % NUM_TEST_SETS, 1'b0, 0, 1'b0);
          q++;
        end
        if (q != held || dut.in_full != 2'b11)
          $display("  [FAIL] Producer waits: %0d sets taken with the consumer holding, expected %0d (staging banks full %b)", q,
                   held, dut.in_full);
        else $display("  [Overrun] %0d sets held; the host had no staging credit for %0d cycles and put nothing", q, WAIT_CYCLES);
        if (FAULT == 1) begin
          // A host that ignores its counter: a set's rows and its put with no staging credit.
          set_files(q % NUM_TEST_SETS, f_a, f_b, f_c);
          fork
            load_west_queue(f_a);
            load_north_queue(f_b);
          join
          @(negedge clk);
          stg.data = '0;
          stg.put  = 1'b1;
          @(negedge clk);
          stg.put = 1'b0;
          repeat (5) @(negedge clk);
          $display("  [Fault] FAULT 1: rows and a staging put with no credit");
          $finish;
        end
        res_hold = 0;
        for (int j = 0; j < q; j++) begin
          set_files(j % NUM_TEST_SETS, f_a, f_b, f_c);
          verify_results(f_c, errs);
          total_sets_run++;
          if (errs == 0) sets_passed++;
          else sets_failed++;
        end
        stage_set(q % NUM_TEST_SETS, 1'b0, 0, 1'b0);  // a set after the wait must land intact
        set_files(q % NUM_TEST_SETS, f_a, f_b, f_c);
        verify_results(f_c, errs);
        total_sets_run++;
        if (errs == 0) sets_passed++;
        else sets_failed++;
      end
      begin
        repeat (TIMEOUT_CYCLES * 6 + WAIT_CYCLES) @(posedge clk);
        $display("  [FATAL] Timeout in the producer-waits test");
        $finish;
      end
    join_any
    disable fork;
    streaming = 0;
    if (count_bad || n_launched != q + 1 || n_completed != q + 1)
      $display("  [FAIL] Producer waits: %0d sets launched and %0d completed, expected %0d each", n_launched,
               n_completed, q + 1);
    drain_check("producer waits");
  endtask

  // ── Weight cache: fills on the region links, cached sets, and a refill that must wait for its region ─────
  // Waits for region r's credit and puts the fill, then writes tiles base.. with B of set t (rev: of set K-1-t), zero padded.
  task automatic fill_region(input int r, input int base, input bit rev, output int waited);
    string f_a, f_b, f_c;
    integer fh, res;
    reg [DATA_WIDTH-1:0] tmp;
    logic [DATA_WIDTH-1:0] q[$];
    waited = 0;
    @(negedge clk);
    while (!wc_has[r]) begin
      @(negedge clk);
      waited++;
    end
    wc_put(r);
    @(posedge clk);  // cache writes are driven after the edge, as the row loaders drive theirs
    for (int t = 0; t < NUM_TEST_SETS; t++) begin
      set_files(rev ? NUM_TEST_SETS - 1 - t : t, f_a, f_b, f_c);
      q.delete();
      fh = $fopen(f_b, "r");
      if (!fh) begin
        $display("  [Error] Could not open NORTH file: %s", f_b);
        $finish;
      end
      while (!$feof(fh)) begin
        res = $fscanf(fh, "%h", tmp);
        if (res == 1) q.push_back(tmp);
      end
      $fclose(fh);
      for (int i = 0; i < SRAM_SIZE; i += HOST_WORDS) begin
        wc_we   = 1'b1;
        wc_addr = WCAW'((base + t) * SRAM_SIZE + i);
        for (int c = 0; c < HOST_WORDS; c++) n_data[c] = (i + c < q.size()) ? q[i+c] : '0;
        @(posedge clk);
      end
    end
    wc_we  = 1'b0;
    n_data = '0;
    @(posedge clk);
  endtask

  task automatic cache_pass();
    int waited;
    $display("\n[STAGE] WEIGHT CACHE: region fills on credit links, %0d cached sets per fill, a refill that waits for its region",
             NUM_TEST_SETS);
    fork
      begin
        fork
          begin : host
            fill_region(0, 0, 1'b0, waited);
            for (int s = 0; s < NUM_TEST_SETS; s++) stage_set(s, 1'b1, s, s == NUM_TEST_SETS - 1);
            if (FAULT == 2) begin
              // A host that writes a region after its last set was put, then stages a set on a region it never filled.
              @(posedge clk);
              wc_we   = 1'b1;
              wc_addr = WCAW'((NUM_TEST_SETS - 1) * SRAM_SIZE);
              @(posedge clk);
              wc_we = 1'b0;
              stage_set(0, 1'b1, HALF, 1'b0);
              repeat (5) @(negedge clk);
              $display("  [Fault] FAULT 2: a cache write into a closed region's staged tile, a cached set on an unfilled region");
              $finish;
            end
            fill_region(0, 0, 1'b1, waited);  // its credit returns only once the region's last set is broadcast
            $display("  [Cache] refill of region 0 waited %0d cycles for its credit", waited);
            if (waited == 0) $display("  [FAIL] Cache: region 0 refilled while its last set was still staged");
            for (int s = 0; s < NUM_TEST_SETS; s++) stage_set(s, 1'b1, NUM_TEST_SETS - 1 - s, s == NUM_TEST_SETS - 1);
            fill_region(1, HALF, 1'b0, waited);
            for (int s = 0; s < NUM_TEST_SETS; s++) stage_set(s, 1'b1, HALF + s, s == NUM_TEST_SETS - 1);
          end
          begin : consumer
            string f_a, f_b, f_c;
            int errs;
            for (int p = 0; p < 3 * NUM_TEST_SETS; p++) begin
              set_files(p % NUM_TEST_SETS, f_a, f_b, f_c);
              verify_results(f_c, errs);
              total_sets_run++;
              if (errs == 0) sets_passed++;
              else sets_failed++;
            end
          end
        join
      end
      begin
        repeat (TIMEOUT_CYCLES * 6) @(posedge clk);
        $display("  [FATAL] Timeout in the weight cache pass");
        $finish;
      end
    join_any
    disable fork;
    drain_check("weight cache");
  endtask

  // ── Top-level stimulus ────────────────────────────────────────────────────
  initial begin
    $dumpfile("TB_SystolicMesh.vcd");
    $dumpvars(0, TB_SystolicMesh);
    if (!$value$plusargs("res_slots=%d", res_slots)) res_slots = (BEATS < RES_CAP) ? BEATS : RES_CAP;
    if (!$value$plusargs("stall_pct=%d", stall_pct)) stall_pct = 0;
    if (res_slots < 1 || res_slots > RES_CAP) begin
      $display("  [FATAL] +res_slots=%0d is outside 1..%0d", res_slots, RES_CAP);
      $finish;
    end

    $display("----------------------------------------------");
    $display(" SYSTOLIC MESH VERIFICATION (BIT-EXACT)       ");
    $display("----------------------------------------------");
    $display(" Matrix Size:    %0d x %0d", MATRIX_SIZE, MATRIX_SIZE);
    $display(" Tile Size:      %0d x %0d", TILE_SIZE, TILE_SIZE);
    $display(" Tiles in mesh:  %0d x %0d", MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
    $display(" Sets to Run:    %0d", NUM_TEST_SETS);
    $display(" Format:         EXP_W=%0d MAN_W=%0d, %0d-bit operands, %0d-bit results", EXP_W, MAN_W, DATA_WIDTH, ACC_W);
    $display(" Result link:    %0d words per beat, %0d beats per set, %0d slots, %0d%% stalls", WIDE_READ, BEATS, res_slots,
             stall_pct);
    $display(" Compare:        bit-exact against mesh_model");
    $display("----------------------------------------------");

    begin
      longint t_serial;
      t_serial = $time;
      for (int i = 0; i < NUM_TEST_SETS; i++) execute_test_set(i);
      $display("  [Serial] %0d sets in %0d cycles", NUM_TEST_SETS, ($time - t_serial) / CLK_PERIOD);
    end
    drain_check("serial");
    stream_all_sets();
    producer_waits_test();
    cache_pass();

    // ── Final report ───────────────────────────────────────────────────────
    $display("\n##############################################");
    $display(" GLOBAL SUMMARY");
    $display("##############################################");
    $display(" Config:         MATRIX=%0d  TILE=%0d", MATRIX_SIZE, TILE_SIZE);
    $display(" Total Sets:     %0d", total_sets_run);
    $display(" Passed Sets:    %0d", sets_passed);
    $display(" Failed Sets:    %0d", sets_failed);
    $display(" Total Elements: %0d", total_elements);
    $display(" Result beats:   %0d framing errors", res_bad);

    // Per-set cycle breakdown
    $display("----------------------------------------------");
    $display(" CYCLE COUNT PER TEST SET");
    $display("----------------------------------------------");
    begin
      longint total_cycles, min_cycles, max_cycles;
      total_cycles = 0;
      min_cycles   = set_cycles[0];
      max_cycles   = set_cycles[0];
      for (int i = 0; i < NUM_TEST_SETS; i++) begin
        $display("  Set %0d : %0d cycles  (%0d ns)", i, set_cycles[i], set_cycles[i] * CLK_PERIOD);
        total_cycles += set_cycles[i];
        if (set_cycles[i] < min_cycles) min_cycles = set_cycles[i];
        if (set_cycles[i] > max_cycles) max_cycles = set_cycles[i];
      end
      $display("----------------------------------------------");
      $display("  Min    : %0d cycles", min_cycles);
      $display("  Max    : %0d cycles", max_cycles);
      $display("  Avg    : %0d cycles", total_cycles / NUM_TEST_SETS);
    end

    $display("##############################################");
    if (sets_failed == 0 && res_bad == 0) $display(" RESULT: SUCCESS");
    else $display(" RESULT: FAILURE");

    $finish;
  end

endmodule
