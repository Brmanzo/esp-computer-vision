// neuron.sv
// Bradley Manzo 2026

module neuron#(
   parameter int unsigned WidthIn     = 1
  ,parameter int unsigned WidthOut    = 2
  ,parameter int unsigned WeightWidth = 2
  ,parameter int unsigned InChannels  = 1

  ,parameter logic signed [InChannels*WeightWidth-1:0] Weights = '0
  ,parameter logic signed [WidthOut-1:0]               Bias    = '0
) (
   input  [InChannels-1:0][WidthIn-1:0] data_i
  ,output signed [WidthOut-1:0]         activation_o
);
  logic signed [WidthOut-1:0] acc_d;

  always_comb begin : neuron_compute
    acc_d = Bias;
    for (int ch = 0; ch < InChannels; ch++) begin
      acc_d = acc_d + WidthOut'(data_i[ch]*Weights[ch*WeightWidth +: WeightWidth]);
    end
  end
  assign activation_o = acc_d;

endmodule
