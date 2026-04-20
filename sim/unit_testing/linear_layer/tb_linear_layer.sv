// tb_linear_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_linear_layer #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 32
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels   = 1
  ,parameter int unsigned OutChannels  = 1
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

`include "injected_weights.vh"
`include "injected_biases.vh"

linear_layer #(
     .InBits     (InBits)
    ,.OutBits    (OutBits)
    ,.WeightBits (WeightBits)
    ,.BiasBits   (BiasBits)
    ,.InChannels  (InChannels)
    ,.OutChannels (OutChannels)
    ,.Weights     (INJECTED_WEIGHTS)
    ,.Biases      (INJECTED_BIASES)
  ) dut (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.valid_i (valid_i)
    ,.ready_o (ready_o)
    ,.data_i  (data_i)
    ,.valid_o (valid_o)
    ,.ready_i (ready_i)
    ,.data_o  (data_o)
  );

endmodule
