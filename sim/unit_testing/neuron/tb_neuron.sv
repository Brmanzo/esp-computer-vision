// tb_neuron.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module tb_neuron #(
   parameter int unsigned InBits     = 1
  ,parameter int unsigned OutBits    = 32
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels = 1
  ,parameter int unsigned GEN_DSP    = 0
)  (
   input  logic                         clk_i
  ,input  logic                         rst_i
  ,input  logic                         en_i
  ,input  logic                         load_bias_i
  ,input  logic [WeightBits-1:0]        weight_i
  ,input  signed [InChannels-1:0][InBits-1:0] data_i
  ,output signed [OutBits-1:0]         data_o
);
  `include "injected_weights_0.vh"
  `include "injected_biases_0.vh"

  wire signed [31:0] acc_full;
  if (GEN_DSP == 0) begin : gen_lut
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
    // For sequential testing, we expect the testbench to toggle en_i and load_bias_i.
    // We also need to select the correct weight for the current channel being processed.
    // Since tb_neuron doesn't know which channel the testbench is driving, we'll
    // just pass the weight through if we had a weight_i port, but for now
    // let's assume the testbench provides the correct weight index.
    
    // Actually, let's add a weight_i port to tb_neuron.
    neuron_dsp #(
         .InBits     (InBits)
        ,.OutBits    (32)
        ,.WeightBits (WeightBits)
        ,.BiasBits   (BiasBits)
      ) dut (
         .clk_i       (clk_i)
        ,.rst_i       (rst_i)
        ,.en_i        (en_i)
        ,.load_bias_i (load_bias_i)
        ,.data_i      (data_i[0])
        ,.weight_i    (weight_i)
        ,.bias_i      (INJECTED_BIASES_0)
        ,.acc_o       (acc_full)
      );
  end

  assign data_o = OutBits'(acc_full);

endmodule
