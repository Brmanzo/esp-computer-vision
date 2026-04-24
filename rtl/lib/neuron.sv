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
   input  logic signed [InChannels-1:0][InBits-1:0] data_i
  ,output logic signed [OutBits-1:0]                data_o
);
  logic signed [InChannels-1:0][OutBits-1:0] addends;
  wire  signed [OutBits-1:0] sum_o;

  generate
    if (InBits == 1) begin : gen_binary
        for (genvar ch = 0; ch < InChannels; ch++) begin : gen_binary_multiply
          assign addends[ch] = data_i[ch][0] ?  OutBits'($signed(Weights[ch*WeightBits +: WeightBits])) : 
                                               -OutBits'($signed(Weights[ch*WeightBits +: WeightBits]));
        end
      // Otherwise multiply normally
      end else begin : gen_normal
        for (genvar ch = 0; ch < InChannels; ch++) begin : gen_normal_multiply
          assign addends[ch] = OutBits'($signed(Weights[ch*WeightBits +: WeightBits])) * $signed(data_i[ch]);
        end
      end
  endgenerate

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
