// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module mac #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned InBits      = 1
  ,parameter  int unsigned OutBits     = 1
  ,parameter  int unsigned AccBits     = 32
  ,parameter  int unsigned WeightBits  = 2
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic [KernelArea-1:0][InBits-1:0] window // 1D Packed Array
  ,input  logic signed [KernelArea-1:0][WeightBits-1:0] weights_i

  ,output logic signed [AccBits-1:0] data_o
);
  logic signed [AccBits-1:0]    acc_d;
  logic signed [WeightBits-1:0] weight;

  always_comb begin
    acc_d = '0;
    for (int i = 0; i < KernelArea; i++) begin
      weight = weights_i[i];
      // Binarized activation encodes a 1 as +1 and a 0 as -1
      if (OutBits == 1) begin
        if (window[i] == 1'b1) begin
          acc_d += AccBits'(weight);
        end else begin
          acc_d -= AccBits'(weight);
        end
      end else begin
        acc_d += (AccBits'(weight) * $signed({1'b0, window[i]}));
      end
    end
  end
  assign data_o = acc_d;

endmodule
