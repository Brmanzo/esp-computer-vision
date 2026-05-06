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

  /* ---------------------------- Multiplication Logic ---------------------------- */
  logic signed [InChannels-1:0][AccBits-1:0] addends;

  generate
    // Binary Input Encoding {0,1} -> {-1,1}, no multiply, just add or subtract weight
    if (InBits == 1) begin : gen_binary_in
      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_binary_activation_multiply
        assign addends[ch] = data_i[ch][0] ?  AccBits'($signed(Weights[ch*WeightBits +: WeightBits])) : 
                                              -AccBits'($signed(Weights[ch*WeightBits +: WeightBits]));
      end
    // Ternary Input Encoding {-1,0,1}, same as binary, but zero input yields zero output
    end else if (InBits == 2) begin : gen_ternary_in
      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_ternary_input_multiply
        wire signed [AccBits-1:0] weight_extended = AccBits'($signed(Weights[ch*WeightBits +: WeightBits]));
        assign addends[ch] = (data_i[ch] == 2'sb01) ?   weight_extended :
                             (data_i[ch] == 2'sb11) ? -$signed(weight_extended) :
                                                       {AccBits{1'b0}};
      end
    // Ternary Weight Encoding {-1,0,1}, same as input, but add or subtract input
    end else if (WeightBits == 2) begin : gen_ternary_weights
      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_ternary_weight_multiply
        wire signed [1:0] w = Weights[ch*WeightBits +: 2];
        // Use an intermediate signed variable to force Icarus into a signed context
        wire signed [AccBits-1:0] data_extended = AccBits'($signed(data_i[ch]));
        assign addends[ch] = (w == 2'sb01) ?   data_extended :
                             (w == 2'sb11) ? -$signed(data_extended) :
                                               {AccBits{1'b0}};
      end
    // Otherwise normal multiplication
    end else begin : gen_normal
      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_normal_multiply
        assign addends[ch] = AccBits'($signed(Weights[ch*WeightBits +: WeightBits]) * $signed(data_i[ch]));
      end
    end
  endgenerate

  /* ------------------------------ Accumulation Logic ------------------------------ */
  logic signed [AccBits-1:0] sum;
  always_comb begin 
    sum = AccBits'(0);
    for (int i = 0; i < InChannels; i++) begin : gen_acc
      sum += addends[i];
    end
  end

  // wire  signed [AccBits-1:0] sum;
  // adder_tree #(
  //    .InBits     (AccBits) // products held at full precision; always > 1 bit
  //   ,.OutBits    (AccBits)
  //   ,.AddendCount(InChannels)
  // ) adder_inst (
  //    .addends_i(addends)
  //   ,.sum_o    (sum)
  // );

  /* ------------------------------ Bias Logic ------------------------------ */
  /* verilator lint_off UNUSEDSIGNAL */
  wire signed [AccBits-1:0] biased_sum;
  /* verilator lint_on UNUSEDSIGNAL */
  assign biased_sum = $signed(sum) + AccBits'($signed(Bias));

  /* ------------------------------ Output Logic ------------------------------ */
  // Binary encoding {-1,1} -> {0,1}, 1 when positive, 0 when negative
  generate
    if (OutBits == 1) begin : gen_binary_out
      assign data_o = OutBits'(biased_sum > AccBits'(0));
    // Ternary encoding {-1,0,1}, 1 when positive, -1 when negative, 0 when zero
    end else if (OutBits == 2) begin : gen_ternary_out
      assign data_o = ( (biased_sum > AccBits'(0)) ? OutBits'(1) :
                        (biased_sum < AccBits'(0)) ? OutBits'(-1) :
                                                     OutBits'(0) );
    end else begin : gen_truncated_out
      assign data_o = biased_sum[(AccBits-1) -: OutBits];
    end
  endgenerate

endmodule
