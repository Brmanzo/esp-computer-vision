// D-type Flip-Flop
// Declared variables are named in accordance with the "BSG Standard Suffixes"
// https://docs.google.com/document/d/1xA5XUzBtz_D6aSyIBQUwFk_kSUdckrfxa2uzGjMgmCU/edit#heading=h.rhtqn8jwjs44
module dff #(
  parameter logic [0:0] ResetVal = 1'b0
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i // positive-polarity, synchronous reset
  ,input  [0:0] d_i
  ,input  [0:0] en_i
  ,output [0:0] q_o
);

  // Internal register.
  logic [0:0] q_r;

  always_ff @(posedge clk_i) begin
    if      (rst_i) q_r <= ResetVal;
    else if (en_i)  q_r <= d_i;
  end

  // Connect the state element to the output
  assign q_o = q_r;
endmodule
