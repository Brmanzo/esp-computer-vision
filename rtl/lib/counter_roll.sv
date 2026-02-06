// counter_roll.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module counter_roll #(
 parameter int unsigned         Width = 8
,parameter logic [Width-1:0] ResetVal = '0
)  (
 input [0:0] clk_i
,input [0:0] rst_i

,input [Width-1:0] max_val_i
,input [0:0]       up_i
,input [0:0]       down_i

,output [Width-1:0] count_o);

logic [Width-1:0] count_q, count_d;
assign count_o = count_q;

wire [0:0] only_up   = up_i && ~down_i;
wire [0:0] only_down = ~up_i && down_i;

always_ff @(posedge clk_i) begin
  // Reset immediately
  if (rst_i) count_q <= ResetVal;
  // Otherwise iterate to next state on positive edge
  else count_q <= count_d;
end

always_comb begin
  count_d = count_q; // Default hold state
  // Reset has highest priority
  if (rst_i) count_d = ResetVal;
  else if (only_up) begin
    // If exclusively up and not saturated, increment
    if (count_q < max_val_i) count_d = count_q + 1'b1;
    // If rolling over, go to zero
    else count_d = '0;
  end else if (only_down) begin
    // If exclusively down and not bottomed out, decrement
    if (count_q > '0) count_d = count_q - 1'b1;
    // If rolling under, go to max
    else count_d = max_val_i;
  // Otherwise hold
  end else count_d = count_q;
end

endmodule
