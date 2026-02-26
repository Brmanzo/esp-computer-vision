// tb_fc_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_neuron #(
   parameter int unsigned WidthIn     = 1
  ,parameter int unsigned WidthOut    = 32
  ,parameter int unsigned WeightWidth = 2
  ,parameter int unsigned BiasWidth   = 8
  ,parameter int unsigned InChannels  = 1
)  (
   input  [InChannels-1:0][WidthIn-1:0] data_i
  ,output signed [WidthOut-1:0]         data_o
);
  `include "injected_weights.vh"
  `include "injected_bias.vh"

neuron #(
     .WidthIn     (WidthIn)
    ,.WidthOut    (WidthOut)
    ,.WeightWidth (WeightWidth)
    ,.BiasWidth   (BiasWidth)
    ,.InChannels  (InChannels)
    ,.Weights     (INJECTED_WEIGHTS)
    ,.Bias        (INJECTED_BIAS)
  ) dut (
     .data_i  (data_i)
    ,.data_o  (data_o)
  );

endmodule
