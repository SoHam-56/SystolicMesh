`timescale 1ns / 100ps

// ProcessingElement in the build's format: every slot of every set against its exact sum, sets back to back, two-pass sets, the extremes.
// int8 sums in wrapping int32 over the whole range; floats take integers of 1 to 255 in magnitude, whose products and sums fp32 holds exactly.
module TB_PE #(
    parameter int K = (sienna_fmt_pkg::EXPONENT_WIDTH == 0) ? 4 : 8  // products per pass; floats K >= ADD_LATENCY + 1, see the note at U
);
  localparam int DW = sienna_fmt_pkg::DATA_WIDTH, ACC_W = sienna_fmt_pkg::ACC_DATA_WIDTH;
  localparam bit IS_INT = sienna_fmt_pkg::EXPONENT_WIDTH == 0;  // the bench's own encoder; the PE never sees the format
  localparam int MAN_W = sienna_fmt_pkg::MANTISSA_WIDTH;
  localparam int BANKS = 3, BW = 2;
  localparam int U = (K < sienna_fmt_pkg::ADD_LATENCY + 1) ? K : sienna_fmt_pkg::ADD_LATENCY + 1;  // int8 2, floats 6; with K < ADD_LATENCY + 1 a back-to-back pass reads a slot whose add is in flight (floats at -GK=4)
  localparam int NSETS = 40;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;

  logic [DW-1:0] a = '0, b = '0;
  logic v = 0, fresh = 0, more = 0, rel = 0;
  logic [BW-1:0] rd_bank = '0;
  logic [U-1:0][ACC_W-1:0] partial;
  logic [BANKS-1:0] fin;

  ProcessingElement #(
      .K(K), .BANKS(BANKS), .U(U), .BW(BW)
  ) dut (
      .clk_i(clk), .rstn_i(rstn), .a_i(a), .b_i(b), .v_i(v), .fresh_i(fresh), .more_i(more), .pack_i(3'b0),
      .a_o(), .b_o(), .v_o(), .fresh_o(), .more_o(), .pack_o(),
      .rd_bank_i(rd_bank), .partial_o(partial), .release_i(rel), .final_o(fin)
  );

  // fp32 bits of an integer below 2^24 in magnitude, built by hand.
  function automatic logic [31:0] fp32_of(input longint x);
    longint m = (x < 0) ? -x : x;
    int e = 0;
    if (m >= (64'sd1 << 24)) $fatal(1, "fp32_of: %0d is outside (-2^24, 2^24), where the encoding is exact", x);
    if (m == 0) return '0;
    while ((m >> (e + 1)) != 0) e++;
    return {x < 0, 8'(127 + e), 23'((m << (23 - e)) & 64'h7FFFFF)};
  endfunction

  // An operand word: int8 its code; floats the integer in the format (exact: at most 8 significant bits).
  function automatic logic [DW-1:0] word(input int x);
    if (IS_INT) return DW'(x);
    return DW'(fp32_of(longint'(x)) >> (23 - MAN_W));
  endfunction

  // A random operand: int8 any code, one $urandom as before; floats a magnitude of 1 to 255 and a sign.
  function automatic int rnd();
    if (IS_INT) return int'($signed(DW'($urandom)));
    return (($urandom % 2) != 0) ? -int'(1 + $urandom % 255) : int'(1 + $urandom % 255);
  endfunction

  // Stimulus and expected slots, all computed before reset is released; product n of a set goes to slot n mod U.
  localparam int LO = IS_INT ? -128 : -255, HI = IS_INT ? 127 : 255;  // the extremes of sets 0 to 2
  int va[NSETS][2*K], vb[NSETS][2*K];
  int npass[NSETS];
  longint sum[NSETS][U];
  logic [ACC_W-1:0] want[NSETS][U];
  int errs = 0, checked = 0;

  initial begin
    for (int s = 0; s < NSETS; s++) begin
      npass[s] = (s == 0 || s % 5 == 3) ? 2 : 1;  // set 0: two passes of LO x LO; in int8 4 products per slot = 65536, beyond int16
      for (int u = 0; u < U; u++) sum[s][u] = 0;
      for (int n = 0; n < npass[s] * K; n++) begin
        case (s)
          0: begin va[s][n] = LO; vb[s][n] = LO; end
          1: begin va[s][n] = LO; vb[s][n] = HI; end
          2: begin va[s][n] = HI; vb[s][n] = HI; end
          default: begin va[s][n] = rnd(); vb[s][n] = rnd(); end
        endcase
        sum[s][n%U] += longint'(va[s][n]) * longint'(vb[s][n]);
      end
      for (int u = 0; u < U; u++) want[s][u] = IS_INT ? ACC_W'(sum[s][u]) : ACC_W'(fp32_of(sum[s][u]));
    end
    repeat (3) @(posedge clk);
    #1 rstn = 1;
    repeat (2) @(posedge clk);
    fork
      begin
        fork
          begin : feeder
            for (int s = 0; s < NSETS; s++)
              for (int p = 0; p < npass[s]; p++)
                for (int kk = 0; kk < K; kk++) begin
                  @(posedge clk);
                  #1 v = 1;
                  a = word(va[s][p*K+kk]);
                  b = word(vb[s][p*K+kk]);
                  fresh = (p == 0);
                  more = (p < npass[s] - 1);
                end
            @(posedge clk);
            #1 v = 0;
            fresh = 0;
            more = 0;
          end
          begin : reader
            for (int s = 0; s < NSETS; s++) begin
              do begin
                @(posedge clk);
                #1;
              end while (!fin[rd_bank]);
              for (int u = 0; u < U; u++) begin
                checked++;
                if (partial[u] !== want[s][u]) begin
                  errs++;
                  $display("[FAIL] set %0d slot %0d: got %h, want %h", s, u, partial[u], want[s][u]);
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
        repeat (20000) @(posedge clk);
        $display("[FATAL] TB_PE timeout");
        $finish;
      end
    join_any
    disable fork;
    $display("TB_PE %s: %0d sets, %0d slots checked, %0d mismatches", sienna_fmt_pkg::FORMAT_NAME, NSETS, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NSETS * U) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
