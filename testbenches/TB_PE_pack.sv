`timescale 1ns / 100ps

// The packed-PE bench in the build's format: NP PEs at spread columns take one stream; each slot must hold only its block's products.
module pe_pack_bench #(
    parameter int K = 16,
    parameter int NP = 4,
    parameter int NSETS = 30,
    parameter int SKIP = 0  // $urandom draws to discard first: B's single build drew fp32's, then bf16's, then int8's operands from one stream
) (
    output logic done_o,
    output int   errs_o,
    output int   checked_o,
    output int   nonzero_o  // expected slots that are not zero: a float bench of zeros checks nothing
);
  localparam int DW = sienna_fmt_pkg::DATA_WIDTH;
  localparam int ACC_W = sienna_fmt_pkg::ACC_DATA_WIDTH;
  localparam int MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH;  // the bench's own encoder; the PE never sees the format
  localparam int ADD1 = sienna_fmt_pkg::ADD_LATENCY + 1;
  localparam int U = (K < ADD1) ? K : ADD1;
  localparam int BANKS = 3, BW = 2, LGK = $clog2(K);
  localparam bit IS_INT = sienna_fmt_pkg::EXPONENT_WIDTH == 0;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;
  logic [DW-1:0] a = '0, b = '0;
  logic v = 0, fresh = 0, rel = 0;
  logic [2:0] pack = '0, pack_q = '0;
  logic [BW-1:0] rd_bank = '0;
  logic [U-1:0][ACC_W-1:0] part[NP];
  logic [BANKS-1:0] fin[NP];
  logic [2:0] pack_o[NP];
  int muls[NP], want_muls[NP];
  int pass_errs = 0;
  always @(posedge clk) pack_q <= pack;

  for (genvar p = 0; p < NP; p++) begin : PE
    ProcessingElement #(.K(K), .BANKS(BANKS), .U(U), .BW(BW),
                        .COL((p * (K - 1)) / (NP - 1))) dut (
        .clk_i(clk), .rstn_i(rstn), .a_i(a), .b_i(b), .v_i(v), .fresh_i(fresh), .more_i(1'b0), .pack_i(pack),
        .a_o(), .b_o(), .v_o(), .fresh_o(), .more_o(), .pack_o(pack_o[p]),
        .rd_bank_i(rd_bank), .partial_o(part[p]), .release_i(rel), .final_o(fin[p]));
    initial muls[p] = 0;
    always @(posedge clk) if (dut.prod_v) muls[p]++;
    always @(negedge clk) if (rstn && pack_o[p] !== pack_q) pass_errs++;  // pack_o is pack_i one cycle later
  end

  // fp32 bits of an integer in [0, 2^24), built by hand: Verilator's $shortrealtobits gave 0, so every float slot read 0 == 0.
  function automatic logic [31:0] fp32_of(input longint v);
    int e = 0;
    if (v < 0 || v >= (64'sd1 << 24)) $fatal(1, "fp32_of: %0d is outside [0, 2^24), where the encoding is exact", v);
    if (v == 0) return '0;
    while ((v >> (e + 1)) != 0) e++;
    return {1'b0, 8'(127 + e), 23'((v << (23 - e)) & 64'h7FFFFF)};
  endfunction

  // Small positive integers are exact in every float format and their sums never cancel; int8 takes the whole range.
  function automatic logic [DW-1:0] word(input int x);
    if (IS_INT) return DW'(x);
    return DW'(fp32_of(longint'(x)) >> (23 - MAN_W));
  endfunction

  int xa[NSETS][K], xb[NSETS][K], sh[NSETS];
  logic [ACC_W-1:0] want[NSETS][NP][U];

  initial begin
    int lst[4];
    done_o = 0;
    errs_o = 0;
    checked_o = 0;
    nonzero_o = 0;
    if (fp32_of(1) != 32'h3F800000 || fp32_of(3) != 32'h40400000 || fp32_of(256) != 32'h43800000 || fp32_of(255) != 32'h437F0000) begin
      errs_o++;
      $display("[FAIL] fp32_of: 1 %h, 3 %h, 256 %h, 255 %h", fp32_of(1), fp32_of(3), fp32_of(256), fp32_of(255));
    end
    repeat (SKIP) void'($urandom);
    lst = '{0, 1, 2, LGK - 1};
    for (int p = 0; p < NP; p++) want_muls[p] = 0;
    for (int s = 0; s < NSETS; s++) begin
      sh[s] = (lst[s % 4] < LGK) ? lst[s % 4] : LGK - 1;
      for (int k = 0; k < K; k++) begin
        xa[s][k] = IS_INT ? int'($urandom_range(0, 255)) - 128 : int'($urandom_range(1, 4));
        xb[s][k] = IS_INT ? int'($urandom_range(0, 255)) - 128 : int'($urandom_range(1, 4));
      end
      for (int p = 0; p < NP; p++) begin
        automatic int col = (p * (K - 1)) / (NP - 1);
        automatic int bw = K >> sh[s];
        automatic int blk = col / bw;
        automatic longint sums[U];
        automatic bit hit[U];
        for (int u = 0; u < U; u++) begin
          sums[u] = 0;
          hit[u] = 0;
        end
        for (int r = 0; r < bw; r++) begin  // the r-th in-block product goes to slot r mod U
          sums[r%U] += longint'(xa[s][blk*bw+r]) * longint'(xb[s][blk*bw+r]);
          hit[r%U] = 1;
        end
        want_muls[p] += bw;
        for (int u = 0; u < U; u++)
          want[s][p][u] = !hit[u] ? '0 : IS_INT ? ACC_W'(sums[u]) : ACC_W'(fp32_of(sums[u]));  // floats: fp32 sums
      end
    end
    begin  // +stim_sum: a checksum of the operands, to show the stream is B's
      automatic int unsigned chk = 0;
      for (int s = 0; s < NSETS; s++) for (int k = 0; k < K; k++) chk = chk * 31 + 32'(xa[s][k]) * 7 + 32'(xb[s][k]);
      if ($test$plusargs("stim_sum")) $display("pe_pack_bench stimulus checksum %08h (K %0d, NP %0d)", chk, K, NP);
    end
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    fork
      begin
        fork
          begin : feeder
            for (int s = 0; s < NSETS; s++)
              for (int k = 0; k < K; k++) begin
                @(posedge clk);
                #1 v = 1;
                a = word(xa[s][k]);
                b = word(xb[s][k]);
                fresh = 1;
                pack = 3'(sh[s]);
              end
            @(posedge clk);
            #1 v = 0;
            fresh = 0;
            pack = '0;
          end
          begin : reader
            for (int s = 0; s < NSETS; s++) begin
              automatic bit all_fin;
              do begin
                @(posedge clk);
                #1;
                all_fin = 1;
                for (int p = 0; p < NP; p++) all_fin &= fin[p][rd_bank];
              end while (!all_fin);
              for (int p = 0; p < NP; p++)
                for (int u = 0; u < U; u++) begin
                  checked_o++;
                  if (want[s][p][u] != '0) nonzero_o++;
                  if (part[p][u] !== want[s][p][u]) begin
                    errs_o++;
                    if (errs_o <= 20) $display("[FAIL] %s set %0d shift %0d PE %0d slot %0d: got %h, want %h", sienna_fmt_pkg::FORMAT_NAME, s,
                                               sh[s], p, u, part[p][u], want[s][p][u]);
                  end
                end
              rel = 1;
              @(posedge clk);
              #1 rel = 0;
              rd_bank = (rd_bank == BW'(BANKS - 1)) ? '0 : rd_bank + 1'b1;
            end
          end
        join
      end
      begin : watchdog
        repeat (40000) @(posedge clk);
        $display("[FATAL] pe_pack_bench %s timeout", sienna_fmt_pkg::FORMAT_NAME);
        $finish;
      end
    join_any
    disable fork;
    for (int p = 0; p < NP; p++)
      if (muls[p] != want_muls[p]) begin
        errs_o++;
        $display("[FAIL] %s PE %0d issued %0d multiplies, want %0d", sienna_fmt_pkg::FORMAT_NAME, p, muls[p], want_muls[p]);
      end
    if (pass_errs != 0) begin
      errs_o++;
      $display("[FAIL] %s pack_o was not pack_i delayed one cycle (%0d cycles)", sienna_fmt_pkg::FORMAT_NAME, pass_errs);
    end
    done_o = 1;
  end
endmodule

// ProcessingElement under packing in the build's format, every shift: fp32 and bf16 K = 16 over 4 PEs (U = 6), int8 K = 8 over 3 (U = 2).
module TB_PE_pack;
  localparam bit IS_INT = sienna_fmt_pkg::EXPONENT_WIDTH == 0;
  localparam int K = IS_INT ? 8 : 16, NP = IS_INT ? 3 : 4, NSETS = 30;
  localparam int U = (K < sienna_fmt_pkg::ADD_LATENCY + 1) ? K : sienna_fmt_pkg::ADD_LATENCY + 1;
  logic d;
  int e, c, z;
  localparam int SKIP = IS_INT ? 2 * 2 * 16 * NSETS : (sienna_fmt_pkg::DATA_WIDTH == 16) ? 2 * 16 * NSETS : 0;  // B's order: fp32, bf16, int8
  pe_pack_bench #(.K(K), .NP(NP), .NSETS(NSETS), .SKIP(SKIP)) X (.done_o(d), .errs_o(e), .checked_o(c), .nonzero_o(z));
  initial begin
    wait (d);
    $display("TB_PE_pack: %s %0d slots (%0d nonzero) %0d errors", sienna_fmt_pkg::FORMAT_NAME, c, z, e);
    if (e == 0 && c == NSETS * NP * U && z * 2 > c) $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
