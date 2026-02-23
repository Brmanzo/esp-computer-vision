// conv_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_conv_layer #(
   parameter  int unsigned LineWidthPx  = 16
  ,parameter  int unsigned LineCountPx  = 12
  ,parameter  int unsigned WidthIn      = 1
  ,parameter  int unsigned WidthOut     = 32
  ,parameter  int unsigned KernelWidth  = 3
  ,parameter  int unsigned WeightWidth  = 2
  ,parameter  int unsigned InChannels   = 1
  ,parameter  int unsigned OutChannels  = 1

  ,parameter  int unsigned Stride            = 1

)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  [InChannels-1:0][WidthIn-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [OutChannels-1:0][WidthOut-1:0] data_o
);

`include "injected_weights.vh"

conv_layer #(
     .LineWidthPx (LineWidthPx)
    ,.LineCountPx (LineCountPx)
    ,.WidthIn     (WidthIn)
    ,.WidthOut    (WidthOut)
    ,.KernelWidth (KernelWidth)
    ,.WeightWidth (WeightWidth)
    ,.InChannels  (InChannels)
    ,.OutChannels (OutChannels)
    ,.Stride      (Stride)
    ,.Weights     (INJECTED_WEIGHTS)
  ) dut (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.valid_i (valid_i)
    ,.ready_o (ready_o)
    ,.data_i  (data_i)
    ,.valid_o (valid_o)
    ,.ready_i (ready_i)
    ,.data_o  (data_o)
  );

endmodule
