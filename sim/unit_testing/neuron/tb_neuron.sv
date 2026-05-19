// tb_neuron.sv
// Bradley Manzo, 2026

/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_neuron #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 32
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned GEN_DSP     = 0
  ,parameter               FileName    = "injected_weights_0.hex"
  ,parameter               FileName_0  = ""
  ,parameter               FileName_hi = "nn/data/roms/hex/zeros.hex"
  ,parameter               FileName_0_hi = ""
)  (
   input  logic                                       clk_i
  ,input  logic                                       rst_i
  ,input  logic                                       valid_i
  ,output logic                                       ready_o
  ,input  logic signed [InChannels-1:0][InBits-1:0]  data_i
  ,input  logic                                       ready_i
  ,output logic                                       valid_o
  ,output logic signed [OutBits-1:0]                  data_o
);
  `include "injected_weights_0.vh"
  `include "injected_biases_0.vh"

  wire signed [31:0] acc_full;

  if (GEN_DSP == 0) begin : gen_lut
    assign ready_o = 1'b1;
    assign valid_o = 1'b1;
    neuron #(
         .InBits     (InBits)
        ,.OutBits    (32)
        ,.WeightBits (WeightBits)
        ,.BiasBits   (BiasBits)
        ,.InChannels (InChannels)
        ,.Weights    (INJECTED_WEIGHTS_0)
        ,.Bias       (INJECTED_BIASES_0)
      ) dut (
         .data_i  (data_i)
        ,.data_o  (acc_full)
      );
  end else begin : gen_dsp
    wire signed [0:0][OutBits-1:0] seq_data_o;
    neuron_seq #(
         .InBits      (InBits)
        ,.OutBits     (OutBits)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.InChannels  (InChannels)
        ,.OutChannels (1)
        ,.DSPCount    (1)
        ,.FileName    ((FileName_0 != "") ? FileName_0 : FileName)
        ,.FileName_hi ((FileName_0_hi != "") ? FileName_0_hi : FileName_hi)
        ,.Biases      (INJECTED_BIASES_0)
      ) dut (
         .clk_i   (clk_i)
        ,.rst_i   (rst_i)
        ,.valid_i (valid_i)
        ,.ready_o (ready_o)
        ,.data_i  (data_i)
        ,.ready_i (ready_i)
        ,.valid_o (valid_o)
        ,.data_o  (seq_data_o)
      );
    assign acc_full = 32'(signed'(seq_data_o[0]));
  end

  assign data_o = OutBits'(acc_full);

endmodule
