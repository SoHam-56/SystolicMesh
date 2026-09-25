`timescale 1ns / 100ps

// Unit test for the pipelined SystolicArray: NSETS random N x K by K x N products streamed back to back.
// The writer loads whenever an operand bank is free and the reader drains whenever a set is final, so sets overlap; each is checked against a real-valued model.
module TB_SystolicArray #(
    parameter int N        = 4,
    parameter int K        = 4,
    parameter int NSETS    = 8,
    parameter int READ_GAP = 0  // idle cycles the reader waits before each set, to exercise back-pressure
);
  localparam int DW = 32;
  localparam int U = (K < 6) ? K : 6;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;

  logic n_we = 0, w_we = 0, commit = 0, rd_en = 0, rel = 0;
  logic [N-1:0][DW-1:0] n_data = '0;
  logic [K-1:0][DW-1:0] w_data = '0;
  logic [$clog2(N*N)-1:0] rd_addr = '0;
  logic [U-1:0][DW-1:0] rd_data;
  logic rd_valid, load_ready, set_final, busy;

  SystolicArray #(
      .N          (N),
      .K          (K),
      .DATA_WIDTH (DW),
      .WEST_WORDS (K),
      .NORTH_WORDS(N),
      .U          (U)
  ) dut (
      .clk_i               (clk),
      .rstn_i              (rstn),
      .north_write_enable_i(n_we),
      .north_write_data_i  (n_data),
      .west_write_enable_i (w_we),
      .west_write_data_i   (w_data),
      .commit_i            (commit),
      .load_ready_o        (load_ready),
      .set_final_o         (set_final),
      .read_enable_i       (rd_en),
      .read_addr_i         (rd_addr),
      .read_data_o         (rd_data),
      .read_valid_o        (rd_valid),
      .release_i           (rel),
      .busy_o              (busy)
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

  logic [DW-1:0] A[NSETS][N][K], B[NSETS][K][N];
  int failed = 0, checked = 0;
  longint cyc = 0, t_commit[NSETS], t_final[NSETS];
  always @(posedge clk) cyc <= cyc + 1;

  // Writer and reader are separate blocks; Verilator's fork/join runtime aborted on the pair inside one fork.
  bit writer_done = 0;
  initial begin
    wait (rstn);
    @(posedge clk) #1;
    for (int s = 0; s < NSETS; s++) begin
      while (!load_ready) @(posedge clk) #1;
      for (int r = 0; r < N; r++) begin
        w_we = 1;
        for (int kk = 0; kk < K; kk++) w_data[kk] = A[s][r][kk];
        if (r < K) begin
          n_we = 1;
          for (int c = 0; c < N; c++) n_data[c] = B[s][r][c];
        end
        @(posedge clk) #1;
      end
      w_we = 0;
      for (int kk = N; kk < K; kk++) begin
        n_we = 1;
        for (int c = 0; c < N; c++) n_data[c] = B[s][kk][c];
        @(posedge clk) #1;
      end
      n_we = 0;
      commit = 1;
      t_commit[s] = cyc;
      @(posedge clk) #1;
      commit = 0;
    end
    writer_done = 1;
  end

  initial begin
    for (int s = 0; s < NSETS; s++) begin
      for (int r = 0; r < N; r++) for (int kk = 0; kk < K; kk++) A[s][r][kk] = rnd();
      for (int kk = 0; kk < K; kk++) for (int c = 0; c < N; c++) B[s][kk][c] = rnd();
    end
    repeat (3) @(posedge clk) #1;
    rstn = 1;
    repeat (2) @(posedge clk) #1;
    for (int s = 0; s < NSETS; s++) begin
      automatic int waited = 0;
      repeat (READ_GAP) @(posedge clk) #1;
      while (!set_final && waited < 5000) begin
        @(posedge clk) #1;
        waited++;
      end
      if (!set_final) begin
        $display("  [FAIL] set %0d never became final", s);
        $display("RESULT: FAILED");
        $finish;
      end
      t_final[s] = cyc;
      for (int p = 0; p < N * N; p++) begin
        automatic real gold = 0.0, mag = 0.0, got = 0.0, err;
        rd_en = 1;
        rd_addr = p[$clog2(N*N)-1:0];
        rel = (p == N * N - 1);
        @(posedge clk) #1;
        rd_en = 0;
        rel = 0;
        for (int kk = 0; kk < K; kk++) begin
          automatic real ab = f32(A[s][p/N][kk]) * f32(B[s][kk][p%N]);
          gold += ab;
          mag += (ab < 0) ? -ab : ab;
        end
        for (int u = 0; u < U; u++) got += f32(rd_data[u]);
        err = (got > gold) ? got - gold : gold - got;
        checked++;
        if (!rd_valid || err > 1.0e-5 * mag + 1.0e-12) begin
          failed++;
          if (failed < 10) $display("  [FAIL] set %0d pixel %0d: got %f expected %f, err %e", s, p, got, gold, err);
        end
      end
    end
    wait (writer_done);
    repeat (5) @(posedge clk) #1;
    if (busy) begin
      failed++;
      $display("  [FAIL] busy_o still high after every set was read");
    end
    $display("SystolicArray N=%0d K=%0d gap=%0d: %0d pixels checked, %0d failed, set 0 commit to final %0d cycles, steady %0d cycles per set",
             N, K, READ_GAP, checked, failed, t_final[0] - t_commit[0],
             (NSETS > 2) ? (t_final[NSETS-1] - t_final[1]) / (NSETS - 2) : 0);
    // A ternary of two strings prints as a number under Verilator, so branch instead.
    if (failed == 0 && checked == NSETS * N * N) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end

  initial begin
    #2000000;
    $display("[FATAL] TB_SystolicArray timeout");
    $display("RESULT: FAILED");
    $finish;
  end
endmodule
