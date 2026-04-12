// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module mac #(
   parameter  int unsigned InBits     = 1
  ,parameter  int unsigned OutBits    = 32
  ,parameter  int unsigned WeightBits = 2
  ,parameter  int unsigned TermCount  = 3
)  (
   input  logic [TermCount-1:0][InBits-1:0]             window_i 
  ,input  logic signed  [TermCount-1:0][WeightBits-1:0] weights_i

  ,output logic signed [OutBits-1:0] sum_o
);

  logic signed [TermCount-1:0][OutBits-1:0] addends; // 2D array to hold the terms at each level of the tree
  
  typedef logic signed [OutBits-1:0] acc_t; 

  generate
    // Multiply
    // If binary input, encode {0,1} as {-1,1} so that multiplication is just +/- the weight
    if (InBits == 1) begin : gen_binary
      for (genvar i = 0; i < TermCount; i++) begin : gen_binary_multiply
        assign addends[i] = window_i[i][0] ? acc_t'(weights_i[i]) : -acc_t'(weights_i[i]);
      end
    // Otherwise multiply normally
    end else begin : gen_normal
      for (genvar i = 0; i < TermCount; i++) begin : gen_normal_multiply
        assign addends[i] = acc_t'(weights_i[i]) * $signed({1'b0, window_i[i]});
      end
    end
  endgenerate

  // Accumulate the products using a balanced adder tree to minimize the critical path
  adder_tree #(
     .InBits     (OutBits) // products are sign extended to output width
    ,.OutBits    (OutBits)
    ,.AddendCount(TermCount)
  ) adder_inst (
     .addends_i(addends)
    ,.sum_o    (sum_o)
  );

endmodule
