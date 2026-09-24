`timescale 1ns / 100ps

module ColumnInputQueue #(
    parameter N          = 8,             // Systolic array dimension
    parameter DATA_WIDTH = 32,
    parameter WRITE_WORDS = 1,  // words per write: 1, or N for one tile row per cycle
    parameter MEM_FILE   = "default.mem"
) (
    input logic clk_i,
    input logic rstn_i,

    input logic         start_i,
    input logic [N-1:0] passthrough_valid_i,

    input logic                  write_enable_i,
    input logic [WRITE_WORDS-1:0][DATA_WIDTH-1:0] write_data_i,
    input logic                  write_reset_i,

    output logic [DATA_WIDTH-1:0] data_o[0:N-1],
    output logic data_valid_o,
    output logic [N-1:0] last_o,

    output logic queue_empty_o
);

  localparam SRAM_DEPTH = N * N;
  localparam COUNT_WIDTH = $clog2(N + 1);

  logic [DATA_WIDTH-1:0] sram[0:SRAM_DEPTH-1];

  logic [$clog2(SRAM_DEPTH):0] write_addr;  // one bit wider: it reaches SRAM_DEPTH after the last write

  logic [$clog2(SRAM_DEPTH)-1:0] read_addr[0:N-1];

  logic passthrough_valid_d1[0:N-1];
  logic passthrough_valid_d2[0:N-1];

  logic queue_active;
  logic first_data_sent;
  logic first_data_pulse;
  logic pe0_data_valid;

  logic [COUNT_WIDTH-1:0] pe_data_count[0:N-1];

  logic last_element_read[0:N-1];
  logic last_element_read_d1[0:N-1];

  initial begin
    if (MEM_FILE != "default.mem") $readmemh(MEM_FILE, sram);
    else for (int i = 0; i < SRAM_DEPTH; i++) sram[i] = '0;
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      write_addr <= '0;
    end else begin
      if (write_reset_i) begin
        write_addr <= '0;
      end else if (write_enable_i && write_addr < SRAM_DEPTH) begin
        for (int c = 0; c < WRITE_WORDS; c++) sram[int'(write_addr)+c] <= write_data_i[c];
        write_addr <= write_addr + WRITE_WORDS;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      for (int i = 0; i < N; i++) begin
        read_addr[i] <= i;
        pe_data_count[i] <= '0;
      end
      queue_active <= 1'b0;
      first_data_sent <= 1'b0;
      first_data_pulse <= 1'b0;
      pe0_data_valid <= 1'b0;
      for (int i = 0; i < N; i++) begin
        last_element_read[i] <= '0;
        last_element_read_d1[i] <= '0;
        passthrough_valid_d1[i] <= '0;
        passthrough_valid_d2[i] <= '0;
      end
    end else begin

      first_data_pulse <= 1'b0;
      pe0_data_valid   <= 1'b0;
      for (int i = 0; i < N; i++) begin
        last_element_read[i] <= '0;

        passthrough_valid_d1[i] <= passthrough_valid_i[i];
        passthrough_valid_d2[i] <= passthrough_valid_d1[i];

        last_element_read_d1[i] <= last_element_read[i];
      end

      if (start_i && !queue_active) begin
        queue_active <= 1'b1;
        first_data_sent <= 1'b0;
        // Read side must return to its reset position. Left alone, pe_data_count
        // stays at N and no element is ever read again.
        for (int i = 0; i < N; i++) begin
          pe_data_count[i] <= '0;
          read_addr[i] <= i;
        end
      end

      if (queue_active && !first_data_sent) begin
        first_data_sent  <= 1'b1;
        first_data_pulse <= 1'b1;
        pe0_data_valid   <= 1'b1;
        for (int i = 0; i < N; i++) pe_data_count[i] <= pe_data_count[i] + 1'b1;
      end

      if (queue_active) begin
        for (int i = 0; i < N; i++) begin
          if (passthrough_valid_d2[i] && pe_data_count[i] < N) begin

            if (pe_data_count[i] == (N - 1)) last_element_read[i] <= 1'b1;

            read_addr[i] <= read_addr[i] + N;
            pe_data_count[i] <= pe_data_count[i] + 1'b1;

            if (i == 0) pe0_data_valid <= 1'b1;
          end
        end
      end

      if (queue_active) begin
        logic all_pes_done;
        all_pes_done = 1'b1;
        for (int i = 0; i < N; i++) if (pe_data_count[i] < N) all_pes_done = 1'b0;
        if (all_pes_done) queue_active <= 1'b0;
      end
    end
  end

  always_comb begin

    for (int i = 0; i < N; i++) data_o[i] = sram[read_addr[i]];

    data_valid_o = pe0_data_valid;
    for (int i = 0; i < N; i++) last_o[i] = last_element_read_d1[i];

    queue_empty_o = 1'b1;
    for (int i = 0; i < N; i++) if (pe_data_count[i] < N) queue_empty_o = 1'b0;
  end

endmodule
