`timescale 1ns / 1ps
module comparator_tree #(
   parameter  int unsigned InBits      = 8,
   parameter  int unsigned OutBits     = InBits,
   parameter  int unsigned ClassCount   = 32,
   parameter  int unsigned IdBits      = (ClassCount <= 1) ? 1 : $clog2(ClassCount),
   localparam int unsigned TreeLevels  = (ClassCount <= 1) ? 1 : $clog2(ClassCount),
   localparam int unsigned TreeWidth   = 1 << TreeLevels
) (
   input  logic signed [ClassCount-1:0][InBits-1:0] classes_i,

   output logic signed [OutBits-1:0] max_o,
   output logic signed [IdBits-1:0]  id_o
);

  typedef logic signed [OutBits-1:0] comp_t;
  typedef logic signed [IdBits-1:0]  id_t;

  // Unpacked array of packed elements
  comp_t tree [TreeLevels:0][TreeWidth-1:0];
  id_t id   [TreeLevels:0][TreeWidth-1:0];

  always_comb begin
    // Initialize Tree as Minimum values, and IDs as 0s
    for (int level = 0; level <= TreeLevels; level++) begin
      for (int idx = 0; idx < TreeWidth; idx++) begin
        tree[level][idx] = {1'b1, {(OutBits-1){1'b0}}};
        id[level][idx]   = '0;
      end
    end

    // Initialize classes at leaves
    for (int i = 0; i < ClassCount; i++) begin
      tree[0][i] = comp_t'(classes_i[i]);
      id[0][i]   = id_t'(i); // Store original index
    end

    // Each successive level of the tree compares pairs of elements from the previous level,
    // effectively halving the number of elements at each level until only one maximum remains at the root
    for (int level = 0; level < TreeLevels; level++) begin
      for (int j = 0; j < TreeWidth; j++) begin
        if (j < (TreeWidth >> (level + 1))) begin
          tree[level+1][j] = tree[level][2*j] >= tree[level][2*j+1] ? tree[level][2*j] : tree[level][2*j+1];
          id[level+1][j]   = tree[level][2*j] >= tree[level][2*j+1] ? id[level][2*j] : id[level][2*j+1];
        end
      end
    end
    max_o    = tree[TreeLevels][0];
    id_o = id[TreeLevels][0];
  end

endmodule
