// balanced_add.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module balanced_add #(
   parameter  int unsigned AccBits     = 32
  ,parameter  int unsigned AddendCount = 32
  ,localparam int unsigned TreeLevels  = $clog2(AddendCount)
  ,localparam int unsigned TreeWidth   = 1 << TreeLevels
)  (
   input  signed [AddendCount-1:0][AccBits-1:0] addends_i // 1D Packed Array

  ,output signed [AccBits-1:0] sum_o
);

  typedef logic signed [AccBits-1:0] acc_t; 

  acc_t tree [TreeLevels:0][TreeWidth-1:0]; // 2D array to hold the terms at each level of the tree

  always_comb begin

    // Load addends into first level of tree, padding to the next power of 2
    for(int i = 0; i < TreeWidth; i++) begin
      if (i < AddendCount) begin
        tree[0][i] = addends_i[i];
      end else begin
        tree[0][i] = '0;
      end
    end

    // Balanced Accumulation Tree (Yosys will optimize away the unused branches)
    for (int i = 0; i < TreeLevels; i++) begin
      for (int j = 0; j < TreeWidth >> (i+1); j++) begin
        // Next layer of the tree sums pairs of the previous layer
        tree[i+1][j] = tree[i][2*j] + tree[i][2*j + 1];
      end
    end    
  end
  assign sum_o = tree[TreeLevels][0];

endmodule
