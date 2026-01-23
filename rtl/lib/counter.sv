`timescale 1ns / 1ps
module counter #(
   parameter int unsigned         Width = 4
  ,parameter logic [Width-1:0] ResetVal = '0
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0] up_i
  ,input [0:0] down_i

  ,output [Width-1:0] count_o
  ,output [Width-2:0] next_count_o
);

  logic [Width-1:0] count_q, count_d;

  assign count_o = count_q;
  assign next_count_o = count_d[Width-2:0];

  wire [0:0] only_up  = (up_i == 1'b1 && down_i == 1'b0);
  wire [0:0] only_down = (up_i == 1'b0 && down_i == 1'b1);

  always_ff @(posedge clk_i) begin
    // Reset immediately
    if (rst_i) count_q <= ResetVal;
    // Otherwise iterate to next state on positive edge
    else         count_q <= count_d;
  end

  always_comb begin
    count_d = count_q;
    // Reset has highest priority
    if (rst_i) count_d = ResetVal;
    // If exclusively up, increment
    else if (only_up) count_d = count_q + 1'b1;
    // If exclusively down, decrement
    else if (only_down) count_d = count_q - 1'b1;
    // Otherwise hold
    else count_d = count_q;
  end

endmodule
