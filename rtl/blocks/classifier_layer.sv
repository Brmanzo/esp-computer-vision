// classifier_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module classifier_layer #(
   parameter  int unsigned TermBits      = 8 // Preserved between global max and comparator tree to maintain precision for comparisons
  ,parameter  int unsigned BusBits     = 8 // Output bus width
  ,parameter  int unsigned TermCount   = 32
  ,parameter  int unsigned ClassCount  = 10
  ,localparam int unsigned IdBits      = (ClassCount <= 1) ? 1 : $clog2(ClassCount)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o

  ,input  signed [ClassCount-1:0][TermBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output [BusBits-1:0] class_o
);

  wire signed [ClassCount-1:0][TermBits-1:0] global_max_o;
  wire [IdBits-1:0] id_o;

  global_max #(
     .InBits     (TermBits)
    ,.TermCount  (TermCount)
    ,.InChannels (ClassCount)
  ) global_max_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.valid_i (valid_i)
    ,.ready_o (ready_o)
    ,.data_i  (data_i)

    ,.valid_o (valid_o)
    ,.ready_i (ready_i)
    ,.data_o  (global_max_o)
  );

  /* verilator lint_off PINCONNECTEMPTY */
  comparator_tree #(
     .InBits     (TermBits)
    ,.ClassCount (ClassCount)
  ) comparator_tree_inst (
     .classes_i (global_max_o)
    ,.max_o     ()
    ,.id_o      (id_o) // Outputs minimum bit width to encode class count
  );

  // Extend class ID to the designated output width of bus
  assign class_o = BusBits'(id_o);

endmodule
