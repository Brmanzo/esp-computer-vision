// tb_neuron.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_neuron #(
   parameter int unsigned InBits     = 1
  ,parameter int unsigned OutBits    = 32
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels  = 1
)  (
   input  [InChannels-1:0][InBits-1:0] data_i
  ,output signed [OutBits-1:0]         data_o
);
  `include "injected_weights.vh"
  `include "injected_bias.vh"

neuron #(
     .InBits     (InBits)
    ,.OutBits    (OutBits)
    ,.WeightBits (WeightBits)
    ,.BiasBits   (BiasBits)
    ,.InChannels (InChannels)
    ,.Weights    (INJECTED_WEIGHTS)
    ,.Bias       (INJECTED_BIAS)
  ) dut (
     .data_i  (data_i)
    ,.data_o  (data_o)
  );

endmodule
