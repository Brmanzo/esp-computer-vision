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
  // Accumulate at full product precision so the adder_tree never receives 1-bit
  // values (which would incorrectly trigger its binary {0,1}->{-1,+1} encoding path).
  // Final truncation to OutBits happens only at data_o.

  logic signed [InChannels-1:0][AccBits-1:0] addends;
  wire  signed [AccBits-1:0] sum;

  generate
    if (InBits == 1) begin : gen_binary
        for (genvar ch = 0; ch < InChannels; ch++) begin : gen_binary_multiply
          assign addends[ch] = data_i[ch][0] ?  AccBits'($signed(Weights[ch*WeightBits +: WeightBits])) : 
                                               -AccBits'($signed(Weights[ch*WeightBits +: WeightBits]));
        end
      // Otherwise multiply normally
      end else begin : gen_normal
        for (genvar ch = 0; ch < InChannels; ch++) begin : gen_normal_multiply
          assign addends[ch] = AccBits'($signed(Weights[ch*WeightBits +: WeightBits]) * $signed(data_i[ch]));
        end
      end
  endgenerate

  adder_tree #(
     .InBits     (AccBits) // products held at full precision; always > 1 bit
    ,.OutBits    (AccBits)
    ,.AddendCount(InChannels)
  ) adder_inst (
     .addends_i(addends)
    ,.sum_o    (sum)
  );

  wire signed [AccBits-1:0] biased_sum;
  assign biased_sum = $signed(sum) + AccBits'($signed(Bias));

  // OutBits==1: binary sign decision matching filter.sv convention (1 if positive, 0 if not)
  // OutBits >1: standard two's complement truncation
  assign data_o = (OutBits == 1) ? OutBits'(biased_sum > AccBits'(0)) // Comparing with '0 forces an unsigned evaluation.
                                 : OutBits'(biased_sum);

endmodule
