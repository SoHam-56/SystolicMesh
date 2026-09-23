`timescale 1ns / 100ps

module TB_SystolicMesh;

  localparam DATA_WIDTH = 32;
  localparam CLK_PERIOD = 10;

  localparam MATRIX_SIZE = 16;
  localparam TILE_SIZE = 4;
  localparam SRAM_SIZE = MATRIX_SIZE * MATRIX_SIZE;

  localparam int NUM_TEST_SETS = 5;

  // Reset once at time zero only. Resetting per set hides every re-arm defect.
  localparam bit B2B_MODE = 1'b1;

  // A matmul is ~120 cycles at N=16; 500k made every hung set a 20-minute wait.
  localparam int TIMEOUT_CYCLES = 20_000;

  // Tolerance Settings
  localparam TOLERANCE_MODE = "RELATIVE";  // "ABSOLUTE", "RELATIVE", or "BOTH"
  localparam real ABS_TOL = 0.001;  // Max absolute difference allowed
  localparam real REL_TOL = 0.01;  // Max relative difference allowed (1%)
  localparam logic ENABLE_TOL = 1'b1;  // 1 = Use tolerance, 0 = Exact match only

  reg clk, rstn, start_mult;

  reg n_we, w_we, n_rst, w_rst;
  reg [DATA_WIDTH-1:0] n_data, w_data;
  wire n_empty, w_empty, complete;
  wire in_ready, coll_complete;
  reg  rel;

  reg                      r_en;
  reg     [          31:0] r_addr;
  wire    [DATA_WIDTH-1:0] r_data;
  wire                     r_valid;

  reg     [DATA_WIDTH-1:0] expected_mem          [0:SRAM_SIZE-1];

  // ── Verification counters ──────────────────────────────────────────────────
  int                      total_sets_run = 0;
  int                      sets_passed = 0;
  int                      sets_failed = 0;
  int                      total_elements = 0;
  int                      tol_pass_elements = 0;

  // ── Cycle-count tracking ───────────────────────────────────────────────────
  // Hardware cycle counter: counts from start pulse until complete asserts.
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
      if (start_mult) begin  // latch start — begin counting next cycle
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
      .DATA_WIDTH (DATA_WIDTH)
  ) dut (
      .clk_i(clk),
      .rstn_i(rstn),
      .start_matrix_mult_i(start_mult),

      .north_write_enable_i(n_we),
      .north_write_data_i  (n_data),
      .north_write_reset_i (n_rst),

      .west_write_enable_i(w_we),
      .west_write_data_i  (w_data),
      .west_write_reset_i (w_rst),

      .north_queue_empty_o(n_empty),
      .west_queue_empty_o(w_empty),
      .matrix_mult_complete_o(complete),
      .collection_complete_o(coll_complete),
      .collection_active_o(),
      .result_release_i(rel),
      .input_ready_o(in_ready),

      .read_enable_i(r_en),
      .read_addr_i  (r_addr),
      .read_data_o  (r_data),
      .read_valid_o (r_valid),
      .wide_read_enable_i(1'b0),
      .wide_read_index_i ('0),
      .wide_read_data_o  (),
      .wide_read_valid_o ()
  );

  // ── Clock ─────────────────────────────────────────────────────────────────
  initial begin
    clk = 0;
    forever #(CLK_PERIOD / 2) clk = ~clk;
  end

  // Manual binary32 decode; $signed() leaves the bit pattern as an integer.
  function automatic real f32(input logic [31:0] b);
    int  e;
    real m, v;
    e = int'(b[30:23]);
    m = real'(longint'(b[22:0])) / 8388608.0;
    if (e == 255) v = 1.0e38;                         // Inf / NaN, clamped so any finite compare fails
    else if (e == 0) v = 0.0;                         // zero / flushed subnormal
    else v = (1.0 + m) * (2.0 ** (e - 127));
    return b[31] ? -v : v;
  endfunction

  // ── Tolerance check ───────────────────────────────────────────────────────
  function automatic logic check_tolerance(
      input [DATA_WIDTH-1:0] expected, input [DATA_WIDTH-1:0] actual, output string tolerance_info);
    real expected_real, actual_real;
    real abs_diff, rel_diff;
    logic abs_ok, rel_ok, result;

    expected_real = f32(expected);
    actual_real = f32(actual);

    abs_diff = (expected_real > actual_real) ?
               (expected_real - actual_real) : (actual_real - expected_real);

    if (expected_real != 0.0)
      rel_diff = abs_diff / ((expected_real > 0) ? expected_real : -expected_real);
    else rel_diff = (actual_real == 0.0) ? 0.0 : 1.0;

    abs_ok = (abs_diff <= ABS_TOL);
    rel_ok = (rel_diff <= REL_TOL);

    case (TOLERANCE_MODE)
      "ABSOLUTE": result = abs_ok;
      "RELATIVE": result = rel_ok;
      "BOTH":     result = abs_ok && rel_ok;
      default:    result = abs_ok;
    endcase

    tolerance_info = $sformatf(
        "Abs=%.4f (Limit %.4f), Rel=%.4f%% (Limit %.2f%%)",
        abs_diff,
        ABS_TOL,
        rel_diff * 100.0,
        REL_TOL * 100.0
    );
    return result;
  endfunction

  // Checker self-test: a loose or broken compare must fail the run before any result is trusted.
  initial begin
    string st_info;
    if (!check_tolerance(32'h3f800000, 32'h3f800003, st_info) ||   // 1.0 vs 1.0 + 3 ulp: pass
        check_tolerance(32'h3f800000, 32'h40000000, st_info) ||    // 1.0 vs 2.0: fail
        check_tolerance(32'h3f800000, 32'h3f7ae148, st_info) ||    // 1.0 vs 0.98: fail
        check_tolerance(32'h3f800000, 32'hbf800000, st_info) ||    // 1.0 vs -1.0: fail
        check_tolerance(32'hbf000000, 32'hbd4ccccd, st_info)) begin // -0.5 vs -0.05: fail
      $display("[FAIL] Tolerance checker self-test failed; results cannot be trusted");
      $finish;
    end
  end

  // ── Reset ─────────────────────────────────────────────────────────────────
  task apply_reset();
    begin
      rstn       = 0;
      start_mult = 0;
      n_we       = 0;
      n_rst      = 0;
      n_data     = 0;
      w_we       = 0;
      w_rst      = 0;
      w_data     = 0;
      r_en       = 0;
      r_addr     = 0;
      rel        = 0;
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
          w_we   = 1;
          w_data = tmp;
          @(posedge clk);
          cnt++;
        end
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
          n_we   = 1;
          n_data = tmp;
          @(posedge clk);
          cnt++;
        end
      end
      n_we = 0;
      @(posedge clk);
      $fclose(fh);
    end
  endtask

  // ── Result verification ───────────────────────────────────────────────────
  task verify_results(input string filename, output int err_count);
    integer fh, i, res;
    reg [DATA_WIDTH-1:0] exp_val, actual_val;
    logic exact_match, tol_match;
    string tol_info;
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

      err_count = 0;
      for (i = 0; i < SRAM_SIZE; i++) begin
        r_en   = 1;
        r_addr = i;
        @(posedge clk);
        while (!r_valid) @(posedge clk);

        actual_val = r_data;
        r_en = 0;
        total_elements++;

        exact_match = (actual_val == expected_mem[i]);

        if (!exact_match && ENABLE_TOL)
          tol_match = check_tolerance(expected_mem[i], actual_val, tol_info);
        else begin
          tol_match = 0;
          tol_info  = "N/A";
        end

        if (exact_match) begin
          // Exact pass — silent
        end else if (tol_match) begin
          $display("    [PASS-TOL] Addr %0d: Exp=0x%h, Act=0x%h | %s", i, expected_mem[i],
                   actual_val, tol_info);
          tol_pass_elements++;
        end else begin
          $display("    [FAIL]     Addr %0d: Exp=0x%h, Act=0x%h", i, expected_mem[i], actual_val);
          if (ENABLE_TOL) $display("               %s", tol_info);
          err_count++;
        end
      end

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

      fork
        load_west_queue(f_a);
        load_north_queue(f_b);
      join

      repeat (10) @(posedge clk);

      // sienna_top refuses to start on an empty queue, so a loaded queue reading empty is a failure.
      load_empty = w_empty || n_empty;
      if (load_empty) $display("  [FAIL] Queue reads empty after load (west=%0b north=%0b)", w_empty, n_empty);

      $display("  [Action] Starting Matrix Mult...");
      start_mult = 1;
      @(posedge clk);
      start_mult = 0;

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
      rel = 1;  // hand the result bank back
      @(posedge clk);
      rel = 0;
      @(posedge clk);

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
  wire mesh_busy = (int'(dut.current_state) != 0) && (int'(dut.current_state) != 7);  // not IDLE, not DONE
  initial forever begin
    @(negedge clk);
    if (streaming) begin
      if ((w_we || n_we) && mesh_busy) in_overlap++;
      if (r_en && mesh_busy) out_overlap++;
      if (int'(dut.current_state) == 1) n_launched++;  // RESET_SEQ lasts one cycle per set
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
            string f_a, f_b, f_c;
            for (int s = 0; s < NUM_TEST_SETS; s++) begin
              set_files(s, f_a, f_b, f_c);
              while (!in_ready) @(posedge clk);
              fork
                load_west_queue(f_a);
                load_north_queue(f_b);
              join
              if (!in_ready) $display("  [FAIL] Start pulsed while input_ready_o is low");
              start_mult = 1;
              @(posedge clk);
              start_mult = 0;
              @(posedge clk);  // let the bank flip land before sampling in_ready again
            end
          end
          begin : consumer
            string f_a, f_b, f_c;
            int errs;
            for (int s = 0; s < NUM_TEST_SETS; s++) begin
              set_files(s, f_a, f_b, f_c);
              while (!coll_complete) @(posedge clk);
              $display("  [Stream] set %0d readable @%0t", s, $time);
              verify_results(f_c, errs);
              total_sets_run++;
              if (errs == 0) sets_passed++;
              else sets_failed++;
              rel = 1;
              @(posedge clk);
              rel = 0;
              @(posedge clk);
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
    $display("  [Stream] consumer reading while mesh busy: %0d cycles", out_overlap);
    if (in_overlap == 0) $display("  [FAIL] Overlap: host never loaded a set while the mesh was busy");
    if (out_overlap == 0) $display("  [FAIL] Overlap: consumer never read a result while the mesh was busy");
    if (count_bad || n_completed != NUM_TEST_SETS || n_launched != NUM_TEST_SETS)
      $display("  [FAIL] %0d sets launched and %0d completed, expected %0d each", n_launched,
               n_completed, NUM_TEST_SETS);
    rel = 1;  // nothing outstanding: must be ignored
    @(posedge clk);
    rel = 0;
    repeat (2) @(posedge clk);
    if (coll_complete) $display("  [FAIL] Release with no result outstanding raised collection_complete_o");
  endtask

  // ── Staging overrun: writes and a start while input_ready_o is low ─────────
  // The consumer withholds release until both result banks and both staging banks are full.
  task automatic staging_overrun_test();
    string f_a, f_b, f_c;
    int errs;
    $display("\n[STAGE] STAGING OVERRUN: queue four sets unreleased, then write and start while not ready");
    n_launched  = 0;
    n_completed = 0;
    count_bad   = 0;
    streaming   = 1;
    fork
      begin
        for (int j = 0; j < 4; j++) begin
          set_files(j % NUM_TEST_SETS, f_a, f_b, f_c);
          while (!in_ready) @(posedge clk);
          fork
            load_west_queue(f_a);
            load_north_queue(f_b);
          join
          start_mult = 1;
          @(posedge clk);
          start_mult = 0;
          @(posedge clk);
        end
        repeat (400) @(posedge clk);  // two results fill both result banks, two sets stay staged
        if (in_ready) $display("  [FAIL] Overrun not reached: input_ready_o high with both banks full");
        set_files(4 % NUM_TEST_SETS, f_a, f_b, f_c);
        fork
          load_west_queue(f_a);
          load_north_queue(f_b);
        join
        start_mult = 1;
        @(posedge clk);
        start_mult = 0;
        repeat (2) @(posedge clk);
        if (int'(dut.ptr_A) != 0 || int'(dut.ptr_B) != 0 || in_ready)
          $display("  [FAIL] Writes or a start were taken while input_ready_o was low: ptr_A=%0d ptr_B=%0d",
                   dut.ptr_A, dut.ptr_B);
        else $display("  [Overrun] 256 writes and a start while not ready were all ignored");
        for (int j = 0; j < 4; j++) begin
          set_files(j % NUM_TEST_SETS, f_a, f_b, f_c);
          while (!coll_complete) @(posedge clk);
          verify_results(f_c, errs);
          total_sets_run++;
          if (errs == 0) sets_passed++;
          else sets_failed++;
          rel = 1;
          @(posedge clk);
          rel = 0;
          @(posedge clk);
        end
        set_files(4 % NUM_TEST_SETS, f_a, f_b, f_c);  // a proper load after the overrun must land intact
        while (!in_ready) @(posedge clk);
        fork
          load_west_queue(f_a);
          load_north_queue(f_b);
        join
        start_mult = 1;
        @(posedge clk);
        start_mult = 0;
        @(posedge clk);
        while (!coll_complete) @(posedge clk);
        verify_results(f_c, errs);
        total_sets_run++;
        if (errs == 0) sets_passed++;
        else sets_failed++;
        rel = 1;
        @(posedge clk);
        rel = 0;
        @(posedge clk);
      end
      begin
        repeat (TIMEOUT_CYCLES * 6) @(posedge clk);
        $display("  [FATAL] Timeout in the staging overrun test");
        $finish;
      end
    join_any
    disable fork;
    streaming = 0;
    if (count_bad || n_launched != 5 || n_completed != 5)
      $display("  [FAIL] Overrun: %0d sets launched and %0d completed, expected 5 each", n_launched,
               n_completed);
  endtask

  // ── Top-level stimulus ────────────────────────────────────────────────────
  initial begin
    $dumpfile("TB_SystolicMesh.vcd");
    $dumpvars(0, TB_SystolicMesh);

    $display("----------------------------------------------");
    $display(" SYSTOLIC MESH VERIFICATION (TOLERANCE MODE)  ");
    $display("----------------------------------------------");
    $display(" Matrix Size:    %0d x %0d", MATRIX_SIZE, MATRIX_SIZE);
    $display(" Tile Size:      %0d x %0d", TILE_SIZE, TILE_SIZE);
    $display(" Tiles in mesh:  %0d x %0d", MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
    $display(" Sets to Run:    %0d", NUM_TEST_SETS);
    $display(" Tolerance Mode: %s", TOLERANCE_MODE);
    if (ENABLE_TOL) begin
      $display(" Abs Tolerance:  %.4f", ABS_TOL);
      $display(" Rel Tolerance:  %.2f%%", REL_TOL * 100.0);
    end else begin
      $display(" Tolerance:      DISABLED (Exact Match Only)");
    end
    $display("----------------------------------------------");

    begin
      longint t_serial;
      t_serial = $time;
      for (int i = 0; i < NUM_TEST_SETS; i++) execute_test_set(i);
      $display("  [Serial] %0d sets in %0d cycles", NUM_TEST_SETS, ($time - t_serial) / CLK_PERIOD);
    end
    stream_all_sets();
    staging_overrun_test();

    // ── Final report ───────────────────────────────────────────────────────
    $display("\n##############################################");
    $display(" GLOBAL SUMMARY");
    $display("##############################################");
    $display(" Config:         MATRIX=%0d  TILE=%0d", MATRIX_SIZE, TILE_SIZE);
    $display(" Total Sets:     %0d", total_sets_run);
    $display(" Passed Sets:    %0d", sets_passed);
    $display(" Failed Sets:    %0d", sets_failed);
    $display(" Total Elements: %0d", total_elements);
    if (ENABLE_TOL)
      $display(
          " Tol Passed Els: %0d (%.1f%%)",
          tol_pass_elements,
          (tol_pass_elements * 100.0) / (total_elements > 0 ? total_elements : 1)
      );

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
    if (sets_failed == 0) $display(" RESULT: SUCCESS");
    else $display(" RESULT: FAILURE");

    $finish;
  end

endmodule
