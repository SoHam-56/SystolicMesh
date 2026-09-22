`timescale 1ns / 100ps

module OutputSram #(
    parameter N = 3,                    // Matrix dimension (NxN)
    parameter DATA_WIDTH = 32,          // Width of each data element
    parameter SRAM_DEPTH = N * N
)(
    input logic clk_i,
    input logic rstn_i,

    // Interface to systolic array
    input logic [DATA_WIDTH-1:0] data_i [0:N-1],
    input logic [N-1:0] drain_i,
    input logic matrix_mult_complete_i,
    input logic rearm_i,                // clears a finished collection for the next matmul

    // SRAM read interface
    input logic read_enable_i,
    input logic [$clog2(SRAM_DEPTH)-1:0] read_addr_i,
    output logic [DATA_WIDTH-1:0] read_data_o,
    output logic read_valid_o,

    // Status signals
    output logic collection_complete_o,
    output logic collection_active_o
);

    logic [DATA_WIDTH-1:0] sram [0:SRAM_DEPTH-1];
    logic sram_write_enable;

    typedef enum logic [1:0] {
        IDLE = 2'b00,
        COLLECTING = 2'b01,
        COMPLETE = 2'b10
    } collection_state_t;
    collection_state_t collection_state, next_collection_state;
    
    logic [$clog2(N)-1:0] current_column, next_current_column;
    logic drain_prev;
    logic drain_pulse;

    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            drain_prev <= 1'b0;
        end else begin
            drain_prev <= |drain_i;
        end
    end
    assign drain_pulse = (|drain_i) & ~drain_prev;

    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (~rstn_i) begin
            collection_state <= IDLE;
            current_column <= N - 1;
            for (integer i = 0; i < SRAM_DEPTH; i++) sram[i] <= '0;
        end else begin
            collection_state <= next_collection_state;
            current_column <= next_current_column;           
            if (sram_write_enable) for (integer row = 0; row < N; row++) sram[row * N + current_column] <= data_i[row];
        end
    end

    always_comb begin

        next_collection_state = collection_state;
        next_current_column = current_column;
        collection_complete_o = 1'b0;
        collection_active_o = 1'b0;
        sram_write_enable = 1'b0;
        
        case (collection_state)
            IDLE: begin
                collection_complete_o = 1'b0;
                collection_active_o = 1'b0;
                next_current_column = N - 1;
                
                if (matrix_mult_complete_i) next_collection_state = COLLECTING;
            end
            
            COLLECTING: begin
                collection_active_o = 1'b1;
                
                if (drain_pulse) begin
                    sram_write_enable = 1'b1;
                    
                    // Move to next column or complete
                    if (current_column == 0) next_collection_state = COMPLETE;
                    else next_current_column = current_column - 1;
                end
            end
            
            COMPLETE: begin
                collection_complete_o = 1'b1;
                collection_active_o = 1'b0;
                // Without this exit the flag sticks until reset and every later
                // matmul sees a collection that already finished.
                if (rearm_i) next_collection_state = IDLE;
            end
            
            default: next_collection_state = IDLE;

        endcase
    end

    // SRAM read interface
    always_ff @(posedge clk_i or negedge rstn_i) begin
        if (!rstn_i) begin
            read_data_o <= '0;
            read_valid_o <= 1'b0;
        end else begin
            if (read_enable_i && read_addr_i < SRAM_DEPTH) begin
                read_data_o <= sram[read_addr_i];
                read_valid_o <= 1'b1;
            end else begin
                read_valid_o <= 1'b0;
            end
        end
    end

endmodule
