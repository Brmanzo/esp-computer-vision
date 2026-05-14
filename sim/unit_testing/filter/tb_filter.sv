// tb_filter.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module tb_filter #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = 1
  ,parameter int unsigned KernelWidth = 3
  ,parameter int unsigned AccBits     = 32
  ,parameter int unsigned Unsigned    = 0
  ,parameter int unsigned ShiftBits   = 0
  ,parameter int unsigned GEN_DSP     = 0
  ,parameter string       FileName    = "injected_weights_0.hex"
  ,parameter string       FileName_0  = ""
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,output [0:0] ready_o
  ,input  [0:0] valid_i
  ,input  logic signed [InChannels-1:0][KernelWidth*KernelWidth-1:0][InBits-1:0] windows_i

  ,input  [0:0] ready_i
  ,output [0:0] valid_o
  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);
  localparam int unsigned KernelArea = KernelWidth * KernelWidth;

  // Include injected weights/biases if needed (for parallel version)
  `include "injected_weights_0.vh"
  `include "injected_biases_0.vh"

  if (GEN_DSP == 0) begin : gen_parallel
    filter #(
       .InBits      (InBits)
      ,.OutBits     (OutBits)
      ,.WeightBits  (WeightBits)
      ,.BiasBits    (BiasBits)
      ,.InChannels  (InChannels)
      ,.KernelWidth (KernelWidth)
      ,.AccBits     (AccBits)
      ,.Unsigned    (Unsigned)
      ,.ShiftBits   (ShiftBits)
      ,.Bias        (INJECTED_BIASES_0) // tb_filter only supports 1 OutChannel for parallel
    ) dut (
       .clk_i     (clk_i)
      ,.rst_i     (rst_i)
      ,.valid_i   (valid_i)
      ,.ready_o   (ready_o)
      ,.windows_i (windows_i)
      ,.weights_i (INJECTED_WEIGHTS_0)
      ,.ready_i   (ready_i)
      ,.valid_o   (valid_o)
      ,.data_o    (data_o[0])
    );
  end else begin : gen_sequential
    filter_seq #(
       .InBits      (InBits)
      ,.OutBits     (OutBits)
      ,.WeightBits  (WeightBits)
      ,.BiasBits    (BiasBits)
      ,.InChannels  (InChannels)
      ,.OutChannels (OutChannels)
      ,.KernelWidth (KernelWidth)
      ,.AccBits     (AccBits)
      ,.Unsigned    (Unsigned)
      ,.ShiftBits   (ShiftBits)
      ,.FileName    (FileName)
      ,.FileName_0  (FileName_0)
      ,.Biases      (INJECTED_BIASES_0)
    ) dut (
       .clk_i     (clk_i)
      ,.rst_i     (rst_i)
      ,.valid_i   (valid_i)
      ,.ready_o   (ready_o)
      ,.windows_i (windows_i)
      ,.ready_i   (ready_i)
      ,.valid_o   (valid_o)
      ,.data_o    (data_o)
    );
  end

endmodule
