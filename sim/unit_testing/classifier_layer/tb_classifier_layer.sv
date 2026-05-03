// tb_linear_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_classifier_layer #(
   parameter  int unsigned TermBits   = 1
  ,parameter  int unsigned TermCount  = 32
  ,parameter  int unsigned BusBits    = 8 // Output bus width
  ,parameter  int unsigned InChannels = 2
  ,parameter  int unsigned ClassCount = 4
  ,parameter  int unsigned WeightBits = 2
  ,parameter  int unsigned BiasBits   = 2
  ,localparam int unsigned IdBits     = (ClassCount <= 1) ? 1 : $clog2(ClassCount)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  signed [InChannels-1:0][TermBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output [BusBits-1:0] class_o
);

`include "injected_weights_0.vh"
`include "injected_biases_0.vh"

classifier_layer #(
     .TermBits    (TermBits)
    ,.TermCount   (TermCount)
    ,.BusBits     (BusBits)
    ,.InChannels  (InChannels)
    ,.ClassCount  (ClassCount)
    
    ,.WeightBits  (WeightBits)
    ,.Weights     (INJECTED_WEIGHTS_0)
    ,.BiasBits    (BiasBits)
    ,.Biases      (INJECTED_BIASES_0)
  ) dut (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.valid_i (valid_i)
    ,.ready_o (ready_o)
    ,.data_i  (data_i)

    ,.valid_o (valid_o)
    ,.ready_i (ready_i)
    ,.class_o (class_o)
  );

endmodule
