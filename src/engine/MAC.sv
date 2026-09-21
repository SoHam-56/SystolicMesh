`timescale 1ns / 100ps

// Accumulating MAC: result_o += data_i * weight_i, one operation per start_i.
//
// The multiply and the accumulate used to run as two blocking states, so each operation cost
// the full multiplier plus adder latency, about 15 cycles, even though only the accumulate is
// loop-carried. The multiply for the next operation does not depend on the accumulator, so it
// now overlaps the current accumulate and the recurrence is the adder alone.
//
// ready_o paces the issue: an accumulate can start only once the previous one has written the
// accumulator back, which is ADD_LAT+1 cycles. Issuing at that rate means a multiply result is
// never waiting on a busy adder, so no result queue is needed.
module MAC #(
    parameter DATA_WIDTH = 32
) (
    input  wire                    clk_i,
    input  wire                    rstn_i,
    input  wire [DATA_WIDTH - 1:0] data_i,
    input  wire [DATA_WIDTH - 1:0] weight_i,
    input  wire                    start_i,
    input  wire                    clear_i,   // zero the accumulator between matmuls
    output reg                     mac_done_o,
    output wire                    ready_o,
    output wire                    busy_o,
    output reg  [DATA_WIDTH - 1:0] result_o
);

  localparam int MUL_LAT = 8;  // fp32Multiplier: valid_i at t, done_o at t+8
  localparam int ADD_LAT = 5;  // fp32Adder: valid_i at t, done_o at t+5
  localparam int MIN_GAP = ADD_LAT + 1;  // accumulator write-back lands a cycle after the add

  reg [DATA_WIDTH-1:0] accumulator;
  reg [DATA_WIDTH-1:0] mul_in1, mul_in2;
  reg                  mul_valid;

  wire [DATA_WIDTH-1:0] mul_result, adder_result;
  wire                  mul_done, add_done;

  // Cycles since the last issue, saturating. Nothing may be issued until the accumulator from
  // the previous operation is back.
  reg [3:0] gap;
  reg       started;

  assign ready_o = ~started | (gap >= MIN_GAP[3:0]);

  // An operation is in flight from the moment it is issued until its accumulate retires. The
  // PE goes idle well before that now, so it needs this to know the accumulator is final.
  reg [1:0] inflight;
  assign busy_o = (inflight != 2'd0);

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) inflight <= 2'd0;
    else
      case ({start_i & ready_o, add_done})
        2'b10:   inflight <= inflight + 2'd1;
        2'b01:   inflight <= inflight - 2'd1;
        default: ;
      endcase
  end

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      gap     <= '0;
      started <= 1'b0;
    end else begin
      if (start_i & ready_o) begin
        gap     <= '0;
        started <= 1'b1;
      end else if (gap != 4'hF) begin
        gap <= gap + 1'b1;
      end
    end
  end

  // Multiply stage: issued straight from the inputs, never blocked by the accumulate.
  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      mul_in1   <= {DATA_WIDTH{1'b0}};
      mul_in2   <= {DATA_WIDTH{1'b0}};
      mul_valid <= 1'b0;
    end else begin
      mul_valid <= start_i & ready_o;
      if (start_i & ready_o) begin
        mul_in1 <= data_i;
        mul_in2 <= weight_i;
      end
    end
  end

  fp32Multiplier MUL (
      .clk_i      (clk_i),
      .rstn_i     (rstn_i),
      .valid_i    (mul_valid),
      .A          (mul_in1),
      .B          (mul_in2),
      .result_o   (mul_result),
      .done_o     (mul_done),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

  // Accumulate stage: fires the cycle a product appears, which by construction is a cycle the
  // adder is free.
  reg [DATA_WIDTH-1:0] add_in2;
  reg                  add_valid;

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      add_in2   <= {DATA_WIDTH{1'b0}};
      add_valid <= 1'b0;
    end else begin
      add_valid <= mul_done;
      if (mul_done) add_in2 <= mul_result;
    end
  end

  fp32Adder ADD (
      .clk_i      (clk_i),
      .rstn_i     (rstn_i),
      .valid_i    (add_valid),
      .A          (accumulator),
      .B          (add_in2),
      .result_o   (adder_result),
      .done_o     (add_done),
      .overflow_o (),
      .underflow_o(),
      .invalid_o  ()
  );

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      accumulator <= {DATA_WIDTH{1'b0}};
      result_o    <= {DATA_WIDTH{1'b0}};
      mac_done_o  <= 1'b0;
    end else begin
      mac_done_o <= add_done;
      // clear_i comes from the drain, which is after the last add of a pass, so it cannot race
      // a result. Without it the accumulator carries into the next matmul.
      if (clear_i) accumulator <= {DATA_WIDTH{1'b0}};
      else if (add_done) accumulator <= adder_result;
      if (add_done) result_o <= adder_result;
    end
  end

endmodule
