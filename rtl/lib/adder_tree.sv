`timescale 1ns / 1ps
module adder_tree #(
   parameter  int unsigned InBits      = 8,
   parameter  int unsigned OutBits     = 32,
   parameter  int unsigned AddendCount = 32,
   localparam int unsigned TreeLevels  = (AddendCount <= 1) ? 1 : $clog2(AddendCount),
   localparam int unsigned TreeWidth   = 1 << TreeLevels
) (
   input  logic signed [AddendCount-1:0][InBits-1:0] addends_i,
   output logic signed [OutBits-1:0] sum_o
);

  typedef logic signed [OutBits-1:0] acc_t;

  // Unpacked array of packed elements
  acc_t tree [TreeLevels:0][TreeWidth-1:0];

  always_comb begin
    // Initialize Tree as all zeros to avoid inferred latches when AddendCount is not a power of two
    for (int level = 0; level <= TreeLevels; level++) begin
      for (int idx = 0; idx < TreeWidth; idx++) begin
        tree[level][idx] = '0;
      end
    end

    // Initialize addends at leaves
    for (int i = 0; i < AddendCount; i++) begin
      if (InBits == 1)
        tree[0][i] = (|addends_i[i]) ? acc_t'(1) : acc_t'(-1); // Map {0,1} -> {-1,+1}
      else
        tree[0][i] = acc_t'(addends_i[i]);
    end

    // Each successive level of the tree sums pairs of elements from the previous level,
    // effectively halving the number of elements at each level until only one sum remains at the root
    for (int level = 0; level < TreeLevels; level++) begin
      for (int j = 0; j < TreeWidth; j++) begin
        if (j < (TreeWidth >> (level + 1))) begin
          tree[level+1][j] = tree[level][2*j] + tree[level][2*j+1];
        end
      end
    end
    sum_o = tree[TreeLevels][0];
  end

endmodule
