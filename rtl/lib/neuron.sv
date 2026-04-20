// neuron.sv
// Bradley Manzo 2026

`timescale 1ns / 1ps
module neuron#(
   parameter int unsigned InBits     = 1
  ,parameter int unsigned OutBits    = 32
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels  = 1

  ,parameter logic signed [InChannels*WeightBits-1:0] Weights = '0
  ,parameter logic signed [BiasBits-1:0]              Bias    = '0
) (
   input  [InChannels-1:0][InBits-1:0] data_i
  ,output signed [OutBits-1:0]         data_o
);
  logic signed [WeightBits-1:0] weight;
  logic signed [InChannels-1:0][OutBits-1:0] addends;
  wire  signed [OutBits-1:0] sum_o;

  always_comb begin : neuron_compute
    for (int ch = 0; ch < InChannels; ch++) begin
      weight = Weights[ch*WeightBits +: WeightBits];
      addends[ch] = (OutBits'(weight) * $signed({1'b0, data_i[ch]}));
    end
  end

  adder_tree #(
     .InBits     (OutBits) // products are sign extended to output width
    ,.OutBits    (OutBits)
    ,.AddendCount(InChannels)
  ) adder_inst (
     .addends_i(addends)
    ,.sum_o    (sum_o)
  );

  assign data_o = sum_o + OutBits'(Bias);

endmodule
