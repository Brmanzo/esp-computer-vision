`timescale 1ns / 1ps
module mac #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WidthIn     = 1
  ,parameter  int unsigned WidthOut    = 32
  ,parameter  int unsigned WeightWidth = 2
  ,localparam int unsigned KernelArea = KernelWidth * KernelWidth
)  (
   input logic [KernelArea-1:0][WidthIn-1:0] window // 1D Packed Array
  ,input  logic signed [KernelArea-1:0][WeightWidth-1:0] weights_i

  ,output logic signed [WidthOut-1:0] data_o
);
  logic signed [WidthOut-1:0]    acc_l;
  logic signed [WeightWidth-1:0] weight_l;
  always_comb begin
    acc_l = '0;
    for (int r = 0; r < KernelWidth; r++) begin
      for (int c = 0; c < KernelWidth; c++) begin
        weight_l = weights_i[r*KernelWidth + c];
        // When binary inputs, only add the weight if the input pixel is a 1
        if (WidthIn-1 == 1) begin // WidthIn includes sign bit, WidthIn = 2 for binary images
          if (window[r*KernelWidth + c] != '0) begin
            acc_l = acc_l + WidthOut'(weight_l);
          end
        end else begin
          acc_l = acc_l + (WidthOut'(weight_l) * $signed({1'b0, window[r*KernelWidth + c]}));
        end
      end
    end
  end
  assign data_o = acc_l;

endmodule
