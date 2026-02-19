// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module max #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WidthIn     = 1
  ,localparam int unsigned WidthOut    = WidthIn
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic [KernelArea-1:0][WidthIn-1:0] window // 1D Packed Array
  ,output logic [WidthOut-1:0] data_o
);
  logic signed [WidthOut-1:0] max;

  always_comb begin
    max = window[0];
    for (int i = 1; i < KernelArea; i++) begin
      if (window[i] > max) max = window[i];
    end
  end
  assign data_o = max;

endmodule
