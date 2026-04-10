// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module mac #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned InBits      = 1
  ,parameter  int unsigned AccBits     = 32
  ,parameter  int unsigned WeightBits  = 2
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic [KernelArea-1:0][InBits-1:0] window // 1D Packed Array
  ,input  logic signed [KernelArea-1:0][WeightBits-1:0] weights_i

  ,output logic signed [AccBits-1:0] data_o
);

  logic signed [KernelArea-1:0][AccBits-1:0] addends; // 2D array to hold the terms at each level of the tree
  
  typedef logic signed [AccBits-1:0] acc_t; 

  always_comb begin

    // Multiply
    // If binary input, encode {0,1} as {-1,1} so that multiplication is just +/- the weight
    if (InBits == 1) begin
      for (int i = 0; i < KernelArea; i++) begin
        addends[i] = window[i] ? acc_t'(weights_i[i]) : -acc_t'(weights_i[i]);
      end
    // Otherwise multiply normally
    end else begin
      for (int i = 0; i < KernelArea; i++) begin
        addends[i] = acc_t'(weights_i[i]) * $signed({1'b0, window[i]});
      end
    end
  end

  // Accumulate the products using a balanced adder tree to minimize the critical path
  balanced_add #(
     .AccBits   (AccBits)
    ,.AddendCount(KernelArea)
  ) adder (
     .addends_i(addends)
    ,.sum_o    (data_o)
  );

endmodule
