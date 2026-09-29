`timescale 1ns / 100ps

// ProcessingElement in int8: every slot of every set against the exact int32 sum, sets back to back, two-pass sets, the extremes.
module TB_PE_int8;
  localparam int EXP_W = 0, MAN_W = 7, DW = 8, ACC_W = 32;
  localparam int K = 4, BANKS = 3, BW = 2;
  localparam int U = 2;  // min(K, add_lat(0, 7) + 1)
  localparam int NSETS = 40;

  logic clk = 0, rstn = 0;
  always #5 clk = ~clk;

  logic [DW-1:0] a = '0, b = '0;
  logic v = 0, fresh = 0, more = 0, rel = 0;
  logic [BW-1:0] rd_bank = '0;
  logic [U-1:0][ACC_W-1:0] partial;
  logic [BANKS-1:0] fin;

  ProcessingElement #(
      .EXP_W(EXP_W), .MAN_W(MAN_W), .DATA_WIDTH(DW), .ACC_W(ACC_W), .K(K), .BANKS(BANKS), .U(U), .BW(BW)
  ) dut (
      .clk_i(clk), .rstn_i(rstn), .a_i(a), .b_i(b), .v_i(v), .fresh_i(fresh), .more_i(more),
      .a_o(), .b_o(), .v_o(), .fresh_o(), .more_o(),
      .rd_bank_i(rd_bank), .partial_o(partial), .release_i(rel), .final_o(fin)
  );

  // Stimulus and expected slots, all computed before reset is released; product n of a set goes to slot n mod U.
  logic signed [DW-1:0] va[NSETS][2*K], vb[NSETS][2*K];
  int npass[NSETS];
  logic [ACC_W-1:0] want[NSETS][U];
  int errs = 0, checked = 0;

  initial begin
    for (int s = 0; s < NSETS; s++) begin
      npass[s] = (s == 0 || s % 5 == 3) ? 2 : 1;  // set 0: two passes of -128 x -128, 4 products per slot = 65536, beyond int16
      for (int u = 0; u < U; u++) want[s][u] = '0;
      for (int n = 0; n < npass[s] * K; n++) begin
        case (s)
          0: begin va[s][n] = 8'h80; vb[s][n] = 8'h80; end
          1: begin va[s][n] = 8'h80; vb[s][n] = 8'h7F; end
          2: begin va[s][n] = 8'h7F; vb[s][n] = 8'h7F; end
          default: begin va[s][n] = DW'($urandom); vb[s][n] = DW'($urandom); end
        endcase
        want[s][n % U] = want[s][n % U] + ACC_W'(longint'(va[s][n]) * longint'(vb[s][n]));
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
              for (int p = 0; p < npass[s]; p++)
                for (int kk = 0; kk < K; kk++) begin
                  @(posedge clk);
                  #1 v = 1;
                  a = va[s][p*K+kk];
                  b = vb[s][p*K+kk];
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
        $display("[FATAL] TB_PE_int8 timeout");
        $finish;
      end
    join_any
    disable fork;
    $display("TB_PE_int8: %0d sets, %0d slots checked, %0d mismatches", NSETS, checked, errs);
    $display("RESULT: %s", (errs == 0 && checked == NSETS * U) ? "PASSED" : "FAILED");
    $finish;
  end
endmodule
