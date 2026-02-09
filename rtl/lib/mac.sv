// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module mac #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WidthIn     = 1
  ,parameter  int unsigned WidthOut    = 32
  ,parameter  int unsigned WeightWidth = 2
  ,localparam int unsigned KernelArea = KernelWidth * KernelWidth
)  (
   input  logic [KernelArea-1:0][WidthIn-1:0] window // 1D Packed Array
  ,input  logic signed [KernelArea-1:0][WeightWidth-1:0] weights_i

  ,output logic signed [WidthOut-1:0] data_o
);
  logic signed [WidthOut-1:0]    acc;
  logic signed [WeightWidth-1:0] weight;

  always_comb begin
    acc = '0;
    for (int i = 0; i < KernelArea; i++) begin
      weight = weights_i[i];
      acc += (WidthOut'(weight) * $signed({1'b0, window[i]}));
    end
  end
  assign data_o = acc;

endmodule
