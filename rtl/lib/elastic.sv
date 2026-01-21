`timescale 1ns / 1ps
module elastic #(
     parameter int unsigned Width = 8
    ,parameter logic [0:0] DatapathGate = 0
    ,parameter logic [0:0] DatapathReset = 0
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [Width - 1:0] data_i
  ,input  [0:0]         valid_i
  ,output [0:0]         ready_o

  ,output [0:0]         valid_o
  ,output [Width - 1:0] data_o
  ,input  [0:0]         ready_i
);

  /* ------------------------------ FSM Logic ------------------------------ */
  typedef enum logic [0:0] {Idle, ValidData} fsm_e;
  fsm_e               state_q, state_d;
  logic [Width - 1:0] data_q,  data_d;
  assign data_o  = data_q;

  /* -------------------------- Output Assignments -------------------------- */
  // Ready when Idle or outputting valid data
  assign ready_o = (state_q == Idle) ||
                   (state_q == ValidData && ready_i);

  // Valid when in ValidData state
  assign valid_o = (state_q == ValidData);

  /* -------------------------- Internal Signals -------------------------- */
  wire [0:0] in_fire =  valid_i && ready_o;
  wire [0:0] stall   = ~valid_i && ready_i;

  /* -------------------------- Current State Logic -------------------------- */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      // Always initialize idle state on reset
      state_q <= Idle;
      // Only reset data if parameter is set
      if (DatapathReset) data_q <= '0;
    end else begin
      state_q <= state_d;
      data_q  <= data_d;
    end
  end

  /* -------------------------- Next State Logic -------------------------- */
  always_comb begin
    state_d = state_q; // Default

    case (state_q)
      Idle: begin
        if (valid_i) state_d = ValidData;
        else         state_d = Idle;
      end
      ValidData: begin
        if (stall) state_d = Idle;
        else       state_d = ValidData;
      end
      default: state_d = Idle;
    endcase
  end

  /* -------------------------- Data Path Logic -------------------------- */
  always_comb begin
    data_d = data_q;
    if (DatapathGate) begin
      if (in_fire) data_d = data_i;
    end else begin
      if (ready_o) data_d = data_i;
    end
  end

endmodule
