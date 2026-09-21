`timescale 1ns / 100ps

module ProcessingElement #(
    parameter DATA_WIDTH = 32
) (
    input wire clk_i,
    input wire rstn_i,

    // Data inputs from neighboring PEs
    input wire [DATA_WIDTH - 1:0] north_i,
    input wire [DATA_WIDTH - 1:0] west_i,

    input wire inputs_valid_i,
    input wire last_element_i,

    input wire select_accumulator_i,  // 1: output accumulator, 0: output data passthrough
    input wire accumulator_valid_i,   // Accumulator content of neighbour indicator

    output reg [DATA_WIDTH - 1:0] south_o,  // Pass data to south PE
    output reg [DATA_WIDTH - 1:0] east_o,   // Muxed: data passthrough OR accumulator output

    output reg passthrough_valid_o,  // Valid for south_o and east_o (passthrough mode)
    output reg fwd_valid_o,          // Same data, released as soon as it is buffered
    output wire accept_o,            // High on the cycle this PE takes its inputs
    output wire idle_o,              // High when this PE has no element in flight
    output reg accumulator_valid_o,  // Valid for east_o when in accumulator mode
    output reg last_element_east_o
);

  reg [DATA_WIDTH - 1:0] buffered_north;
  reg [DATA_WIDTH - 1:0] buffered_west;
  reg [DATA_WIDTH - 1:0] buffered_accumulator;

  wire [DATA_WIDTH - 1:0] mac_result;
  wire mac_done;
  wire mac_ready;
  wire mac_busy;
  reg mac_start;

  wire select_accumulator_gated;

  reg last_element_captured;  // Track last element processing

  reg accumulator_drain_flag; // Track if we came to OUTPUT state from IDLE due to accumulator_valid_i

  wire last_element_pulse;
  assign last_element_pulse = mac_done & last_element_captured;

  // FORWARD releases the passthrough operands to the neighbours before the MAC finishes.
  // Neither south_o nor east_o depends on mac_result, so holding them until OUTPUT made the
  // whole wavefront advance one PE per MAC instead of one PE per hop.
  typedef enum reg [2:0] {
    IDLE        = 3'b000,
    LOAD_DATA   = 3'b001,
    FORWARD     = 3'b010,
    MAC_COMPUTE = 3'b011,
    OUTPUT      = 3'b100
  } state_t;

  state_t current_state, next_state;

  assign select_accumulator_gated = select_accumulator_i & (current_state == IDLE);

  // The upstream join holds its valid until accept_o fires, so a forward is never dropped while
  // this PE is busy. idle_o lets the mesh hold the drain wave until every PE has finished,
  // which the wave's one-cycle-per-column sweep otherwise assumes.
  assign accept_o = (current_state == IDLE) & inputs_valid_i;
  assign idle_o   = (current_state == IDLE) & ~mac_busy;  // the accumulator must be final too

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      current_state <= IDLE;
    end else begin
      current_state <= next_state;
    end
  end

  always @(*) begin
    case (current_state)
      IDLE: begin
        if (inputs_valid_i) next_state = LOAD_DATA;
        else if (accumulator_valid_i) next_state = OUTPUT;
        else next_state = IDLE;
      end
      LOAD_DATA: begin
        next_state = FORWARD;
      end
      FORWARD: begin
        next_state = MAC_COMPUTE;
      end
      MAC_COMPUTE: begin
        if (mac_ready) next_state = OUTPUT;
        else next_state = MAC_COMPUTE;
      end
      OUTPUT: begin
        next_state = IDLE;
      end
      default: next_state = IDLE;
    endcase
  end

  // Track the transition from IDLE to OUTPUT due to accumulator_valid_i
  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      accumulator_drain_flag <= 1'b0;
    end else begin
      if (current_state == IDLE && accumulator_valid_i && next_state == OUTPUT) begin
        accumulator_drain_flag <= 1'b1;
      end else if (current_state == OUTPUT) begin
        accumulator_drain_flag <= 1'b0;  // Clear after OUTPUT state
      end
    end
  end

  // Last element capture logic
  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      last_element_captured <= 1'b0;
    end else begin
      // Capture last_element_i pulse (independent of FSM state)
      if (last_element_i) begin
        last_element_captured <= 1'b1;
      end

      // Clear the captured flag when last element pulse is generated
      if (last_element_pulse) begin
        last_element_captured <= 1'b0;  // Clear for next operation
      end
    end
  end

  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      south_o <= {DATA_WIDTH{1'b0}};
      east_o <= {DATA_WIDTH{1'b0}};

      buffered_north <= {DATA_WIDTH{1'b0}};
      buffered_west <= {DATA_WIDTH{1'b0}};

      passthrough_valid_o <= 1'b0;
      fwd_valid_o <= 1'b0;
      accumulator_valid_o <= 1'b0;
      mac_start <= 1'b0;
    end else begin
      case (current_state)
        IDLE: begin
          mac_start <= 1'b0;
          passthrough_valid_o <= 1'b0;
          fwd_valid_o <= 1'b0;

          // Handle accumulator draining in IDLE state
          if (select_accumulator_gated) begin
            east_o <= buffered_accumulator;
            accumulator_valid_o <= 1'b1;
          end else begin
            accumulator_valid_o <= 1'b0;
          end
        end

        LOAD_DATA: begin
          buffered_north <= north_i;
          buffered_west <= west_i;

          mac_start <= 1'b1;
          passthrough_valid_o <= 1'b0;
          fwd_valid_o <= 1'b0;
          accumulator_valid_o <= 1'b0;
        end

        // Hand the operands on now; the MAC keeps running behind them.
        FORWARD: begin
          south_o <= buffered_north;
          east_o <= buffered_west;
          fwd_valid_o <= 1'b1;

          mac_start <= 1'b0;
          passthrough_valid_o <= 1'b0;
          accumulator_valid_o <= 1'b0;
        end

        MAC_COMPUTE: begin

          mac_start <= 1'b0;
          passthrough_valid_o <= 1'b0;
          fwd_valid_o <= 1'b0;
          accumulator_valid_o <= 1'b0;

        end

        OUTPUT: begin
          south_o <= buffered_north;
          fwd_valid_o <= 1'b0;

          if (accumulator_drain_flag) begin
            accumulator_valid_o <= 1'b1;
            passthrough_valid_o <= 1'b0;
            east_o <= west_i;
          end else begin
            passthrough_valid_o <= 1'b1;
            accumulator_valid_o <= 1'b0;
            east_o <= buffered_west;
          end
        end

        default: begin
          mac_start <= 1'b0;
          fwd_valid_o <= 1'b0;
        end
      endcase
    end
  end

  // Results now land after the FSM has moved on, so these follow mac_done rather than a state.
  always @(posedge clk_i or negedge rstn_i) begin
    if (!rstn_i) begin
      buffered_accumulator <= {DATA_WIDTH{1'b0}};
      last_element_east_o  <= 1'b0;
    end else begin
      last_element_east_o <= last_element_pulse;
      if (mac_done) buffered_accumulator <= mac_result;
    end
  end

  MAC #(
      .DATA_WIDTH(DATA_WIDTH)
  ) MAC_UNIT (
      .clk_i(clk_i),
      .rstn_i(rstn_i),
      .data_i(west_i),
      .weight_i(north_i),
      .start_i(mac_start),
      .mac_done_o(mac_done),
      .ready_o(mac_ready),
      .busy_o(mac_busy),
      .result_o(mac_result)
  );

endmodule
