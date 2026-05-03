`timescale 1ns / 1ps
module comparator_tree #(
   parameter  int unsigned InBits      = 8,
   parameter  int unsigned OutBits     = InBits,
   parameter  int unsigned ClassCount  = 32,
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
    // Initialize Tree: min signed for multi-bit, '0 (min unsigned) for 1-bit
    for (int level = 0; level <= TreeLevels; level++) begin
      for (int idx = 0; idx < TreeWidth; idx++) begin
        tree[level][idx] = (InBits == 1) ? '0 : {1'b1, {(OutBits-1){1'b0}}};
        id[level][idx]   = '0;
      end
    end

    // Initialize classes at leaves (no re-encoding; Python model uses raw values)
    for (int i = 0; i < ClassCount; i++) begin
      tree[0][i] = comp_t'(classes_i[i]);
      id[0][i]   = id_t'(i);
    end

    // Each successive level compares pairs; for 1-bit use unsigned bit[0] comparison
    // to correctly order {0,1} (0 < 1 unsigned) instead of {0,-1} (signed 1-bit)
    for (int level = 0; level < TreeLevels; level++) begin
      for (int j = 0; j < TreeWidth; j++) begin
        if (j < (TreeWidth >> (level + 1))) begin
          if (InBits == 1) begin
            // Unsigned 1-bit: bit=1 beats bit=0; left wins on tie (>=)
            if (tree[level][2*j][0] >= tree[level][2*j+1][0]) begin
              tree[level+1][j] = tree[level][2*j];
              id[level+1][j]   = id[level][2*j];
            end else begin
              tree[level+1][j] = tree[level][2*j+1];
              id[level+1][j]   = id[level][2*j+1];
            end
          end else begin
            tree[level+1][j] = tree[level][2*j] >= tree[level][2*j+1] ? tree[level][2*j] : tree[level][2*j+1];
            id[level+1][j]   = tree[level][2*j] >= tree[level][2*j+1] ? id[level][2*j] : id[level][2*j+1];
          end
        end
      end
    end
    max_o    = tree[TreeLevels][0];
    id_o = id[TreeLevels][0];
  end

endmodule
