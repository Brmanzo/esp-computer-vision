// neuron.sv
// Bradley Manzo 2026

module neuron#(
   parameter int unsigned WidthIn     = 1
  ,parameter int unsigned WidthOut    = 32
  ,parameter int unsigned WeightWidth = 2
  ,parameter int unsigned BiasWidth   = 8
  ,parameter int unsigned InChannels  = 1

  ,parameter logic signed [InChannels*WeightWidth-1:0] Weights = '0
  ,parameter logic signed [BiasWidth-1:0]              Bias    = '0
) (
   input  [InChannels-1:0][WidthIn-1:0] data_i
  ,output signed [WidthOut-1:0]         data_o
);
  logic signed [WidthOut-1:0]    acc_d;
  logic signed [WeightWidth-1:0] weight;

  always_comb begin : neuron_compute
    acc_d = '0;
    for (int ch = 0; ch < InChannels; ch++) begin
      weight = Weights[ch*WeightWidth +: WeightWidth];
      acc_d += (WidthOut'(weight) * $signed({1'b0, data_i[ch]}));
    end
    acc_d += WidthOut'(Bias);
  end
  assign data_o = acc_d;

endmodule
