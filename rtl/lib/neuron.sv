// neuron.sv
// Bradley Manzo 2026

`timescale 1ns / 1ps
module neuron#(
   parameter int unsigned InBits     = 1
  ,parameter int unsigned OutBits    = 32
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels = 1

  ,parameter logic signed [InChannels*WeightBits-1:0] Weights = '0
  ,parameter logic signed [BiasBits-1:0]              Bias    = '0
  // If requested OutBits is small, accumulate at full precision before truncating
  ,localparam int unsigned AccBits   = (OutBits > WeightBits + InBits) ? OutBits : WeightBits + InBits
) (
   input  logic signed [InChannels-1:0][InBits-1:0] data_i
  ,output logic signed [OutBits-1:0]                data_o
);

  /* ---------------------------- MAC Logic ---------------------------- */
  wire signed [AccBits-1:0] mac_sum;
  mac #(
     .InBits    (InBits)
    ,.OutBits   (AccBits)
    ,.WeightBits(WeightBits)
    ,.TermCount (InChannels)
  ) mac_inst (
     .window_i (data_i)
    ,.weights_i(Weights)
    ,.sum_o    (mac_sum)
  );

  /* ------------------------------ Bias Logic ------------------------------ */
  wire signed [AccBits-1:0] biased_sum;
  assign biased_sum = mac_sum + AccBits'($signed(Bias));

  /* ------------------------------ Output Logic ------------------------------ */
  // Raw output to match neuron_dsp.sv behavior
  assign data_o = OutBits'($signed(biased_sum));

endmodule
