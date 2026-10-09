`timescale 1ns / 100ps

// One format's packed-PE bench: NP PEs at spread columns take one stream; each slot must hold only its block's products.
module pe_pack_bench #(
    parameter int EXP_W = 8,
    parameter int MAN_W = 23,
    parameter int K = 16,
    parameter int NP = 4,
    parameter int NSETS = 30
) (
    output logic done_o,
    output int   errs_o,
    output int   checked_o,
    output int   nonzero_o  // expected slots that are not zero: a float bench of zeros checks nothing
);
  localparam int DW = 1 + EXP_W + MAN_W;
  localparam int ACC_W = sienna_fmt_pkg::acc_w(EXP_W, MAN_W);
  localparam int ADD1 = sienna_fmt_pkg::add_lat(EXP_W, MAN_W) + 1;
  localparam int U = (K < ADD1) ? K : ADD1;
  localparam int BANKS = 3, BW = 2, LGK = $clog2(K);
  localparam bit IS_INT = sienna_fmt_pkg::is_int(EXP_W);

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
    ProcessingElement #(.EXP_W(EXP_W), .MAN_W(MAN_W), .DATA_WIDTH(DW), .ACC_W(ACC_W), .K(K), .BANKS(BANKS), .U(U), .BW(BW),
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
          want[s][p][u] = !hit[u] ? '0 : IS_INT ? ACC_W'(sums[u]) : ACC_W'(fp32_of(sums[u]) >> (23 - MAN_W));
      end
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
                    if (errs_o <= 20) $display("[FAIL] EXP_W=%0d set %0d shift %0d PE %0d slot %0d: got %h, want %h", EXP_W, s,
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
        $display("[FATAL] pe_pack_bench EXP_W=%0d MAN_W=%0d timeout", EXP_W, MAN_W);
        $finish;
      end
    join_any
    disable fork;
    for (int p = 0; p < NP; p++)
      if (muls[p] != want_muls[p]) begin
        errs_o++;
        $display("[FAIL] EXP_W=%0d PE %0d issued %0d multiplies, want %0d", EXP_W, p, muls[p], want_muls[p]);
      end
    if (pass_errs != 0) begin
      errs_o++;
      $display("[FAIL] EXP_W=%0d pack_o was not pack_i delayed one cycle (%0d cycles)", EXP_W, pass_errs);
    end
    done_o = 1;
  end
endmodule

// ProcessingElement under packing in fp32 (U = 6), bf16 (U = 6) and int8 (U = 2), K = 16 / 16 / 8, every shift.
module TB_PE_pack;
  logic d_f, d_b, d_i;
  int e_f, e_b, e_i, c_f, c_b, c_i, z_f, z_b, z_i;
  pe_pack_bench #(.EXP_W(8), .MAN_W(23), .K(16), .NP(4)) F (.done_o(d_f), .errs_o(e_f), .checked_o(c_f), .nonzero_o(z_f));
  pe_pack_bench #(.EXP_W(8), .MAN_W(7), .K(16), .NP(4)) B (.done_o(d_b), .errs_o(e_b), .checked_o(c_b), .nonzero_o(z_b));
  pe_pack_bench #(.EXP_W(0), .MAN_W(7), .K(8), .NP(3)) I (.done_o(d_i), .errs_o(e_i), .checked_o(c_i), .nonzero_o(z_i));
  initial begin
    wait (d_f && d_b && d_i);
    $display("TB_PE_pack: fp32 %0d slots (%0d nonzero) %0d errors, bf16 %0d slots (%0d nonzero) %0d errors, int8 %0d slots (%0d nonzero) %0d errors",
             c_f, z_f, e_f, c_b, z_b, e_b, c_i, z_i, e_i);
    if (e_f + e_b + e_i == 0 && c_f == 30 * 4 * 6 && c_b == 30 * 4 * 6 && c_i == 30 * 3 * 2 && z_f * 2 > c_f && z_b * 2 > c_b && z_i * 2 > c_i)
      $display("RESULT: PASSED");
    else $display("RESULT: FAILED");
    $finish;
  end
endmodule
