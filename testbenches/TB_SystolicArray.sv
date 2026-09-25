`timescale 1ns / 100ps

// Unit test for SystolicArray: random N x K by K x N products, several sets back to back, each checked against a real-valued model.
module TB_SystolicArray #(
    parameter int N     = 4,
    parameter int K     = 4,
    parameter int NSETS = 6
);
  localparam int DW = 32;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;

  logic start = 0, rearm = 0;
  logic n_we = 0, w_we = 0, n_rst = 0, w_rst = 0;
  logic [N-1:0][DW-1:0] n_data = '0;
  logic [K-1:0][DW-1:0] w_data = '0;
  logic rd_en = 0;
  logic [$clog2(N*N)-1:0] rd_addr = '0;
  logic [DW-1:0] rd_data;
  logic rd_valid, complete, active, mm_done, n_empty, w_empty;

  SystolicArray #(
      .N          (N),
      .K          (K),
      .DATA_WIDTH (DW),
      .WEST_WORDS (K),
      .NORTH_WORDS(N)
  ) dut (
      .clk_i                 (clk),
      .rstn_i                (rstn),
      .start_matrix_mult_i   (start),
      .rearm_i               (rearm),
      .north_write_enable_i  (n_we),
      .north_write_data_i    (n_data),
      .north_write_reset_i   (n_rst),
      .west_write_enable_i   (w_we),
      .west_write_data_i     (w_data),
      .west_write_reset_i    (w_rst),
      .north_queue_empty_o   (n_empty),
      .west_queue_empty_o    (w_empty),
      .matrix_mult_complete_o(mm_done),
      .read_enable_i         (rd_en),
      .read_addr_i           (rd_addr),
      .read_data_o           (rd_data),
      .read_valid_o          (rd_valid),
      .collection_complete_o (complete),
      .collection_active_o   (active)
  );

  function automatic real f32(input logic [31:0] b);
    int e;
    real m, v;
    e = int'(b[30:23]);
    m = real'(longint'(b[22:0])) / 8388608.0;
    if (e == 255) v = 1.0e38;
    else if (e == 0) v = 0.0;
    else v = (1.0 + m) * (2.0 ** (e - 127));
    return b[31] ? -v : v;
  endfunction

  // Random fp32 with magnitude in [2^-7, 2): varied exponents, random sign and mantissa.
  function automatic logic [31:0] rnd();
    logic [7:0] e;
    e = 8'(120 + ($urandom % 8));
    return {1'($urandom), e, 23'($urandom)};
  endfunction

  logic [DW-1:0] A[N][K], B[K][N];
  int failed = 0, checked = 0, cycles, worst_cycles = 0;

  initial begin
    repeat (3) @(posedge clk) #1;
    rstn = 1;
    repeat (2) @(posedge clk) #1;
    for (int set = 0; set < NSETS; set++) begin
      for (int r = 0; r < N; r++) for (int kk = 0; kk < K; kk++) A[r][kk] = rnd();
      for (int kk = 0; kk < K; kk++) for (int c = 0; c < N; c++) B[kk][c] = rnd();
      // Re-arm and rewind as the mesh does, then write one row per cycle.
      rearm = 1; n_rst = 1; w_rst = 1;
      @(posedge clk) #1;
      rearm = 0; n_rst = 0; w_rst = 0;
      @(posedge clk) #1;
      if (set > 0 && complete) begin
        failed++;
        $display("  [FAIL] set %0d: collection_complete_o still high after re-arm", set);
      end
      for (int r = 0; r < N; r++) begin
        w_we = 1;
        for (int kk = 0; kk < K; kk++) w_data[kk] = A[r][kk];
        @(posedge clk) #1;
      end
      w_we = 0;
      for (int kk = 0; kk < K; kk++) begin
        n_we = 1;
        for (int c = 0; c < N; c++) n_data[c] = B[kk][c];
        @(posedge clk) #1;
      end
      n_we = 0;
      start = 1;
      @(posedge clk) #1;
      start = 0;
      cycles = 1;
      if (complete) begin
        failed++;
        $display("  [FAIL] set %0d: complete the cycle after start", set);
      end
      while (!complete && cycles < 5000) begin
        @(posedge clk) #1;
        cycles++;
      end
      if (!complete) begin
        failed++;
        $display("  [FAIL] set %0d: no completion in 5000 cycles", set);
        break;
      end
      if (cycles > worst_cycles) worst_cycles = cycles;
      for (int p = 0; p < N * N; p++) begin
        automatic real gold = 0.0, mag = 0.0, got, err;
        rd_en = 1;
        rd_addr = p[$clog2(N*N)-1:0];
        @(posedge clk) #1;
        rd_en = 0;
        @(negedge clk);
        for (int kk = 0; kk < K; kk++) begin
          gold += f32(A[p/N][kk]) * f32(B[kk][p%N]);
          mag += (f32(A[p/N][kk]) * f32(B[kk][p%N])) < 0 ? -(f32(A[p/N][kk]) * f32(B[kk][p%N])) : f32(A[p/N][kk]) * f32(B[kk][p%N]);
        end
        got = f32(rd_data);
        err = (got > gold) ? got - gold : gold - got;
        checked++;
        if (!rd_valid || err > 1.0e-5 * mag + 1.0e-12) begin
          failed++;
          if (failed < 10)
            $display("  [FAIL] set %0d pixel %0d: got %h (%f) expected %f, err %e, sum|ab| %f", set, p, rd_data, got, gold, err, mag);
        end
      end
      $display("  set %0d: %0d cycles from start to complete", set, cycles);
    end
    $display("SystolicArray N=%0d K=%0d: %0d pixels checked, %0d failed, worst %0d cycles", N, K, checked, failed, worst_cycles);
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (failed == 0 && checked == NSETS * N * N) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
