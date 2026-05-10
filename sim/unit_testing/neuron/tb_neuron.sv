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
  ,parameter string       FileName   = "injected_weights_0.hex"
)  (
   input  logic                         clk_i
  ,input  logic                         rst_i
  ,input  logic                         en_i
  ,input  logic                         load_bias_i
  ,input  logic [WeightBits-1:0]        weight_i // Kept for backward compatibility/manual overrides
  ,input  signed [InChannels-1:0][InBits-1:0] data_i
  ,output signed [OutBits-1:0]         data_o
);
  `include "injected_weights_0.vh"
  `include "injected_biases_0.vh"

  wire signed [31:0] acc_full;
  // If testing comb neuron, still input weights from parameter
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
    // Sequential accumulation logic
    // Drive ROM address with look-ahead to avoid race conditions with DSP sampling
    localparam int unsigned AddrBits = (InChannels > 1) ? $clog2(InChannels) : 1;
    logic [AddrBits-1:0] rom_addr;
    wire  [AddrBits-1:0] next_rom_addr = (rst_i || load_bias_i) ? '0 :
                                         (en_i) ? (rom_addr + 1'b1) :
                                         rom_addr;

    always_ff @(posedge clk_i) begin
      if (rst_i || load_bias_i) rom_addr <= '0;
      else if (en_i)            rom_addr <= rom_addr + 1'b1;
    end

    wire [WeightBits-1:0] rom_weight;
    multi_weight_rom #(
       .WeightBits (WeightBits)
      ,.WeightCount(InChannels)
      ,.FileName   (FileName)
    ) weight_rom_inst (
       .clk_i     (clk_i)
      ,.rst_i     (rst_i)
      ,.rd_addr_i (next_rom_addr) // Look-ahead
      ,.weight_o  (rom_weight)
    );

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
        ,.data_i      (data_q_in)
        ,.weight_i    (rom_weight)
        ,.bias_i      (INJECTED_BIASES_0)
        ,.acc_o       (acc_full)
      );

      // Simple input mux based on look-ahead channel
      wire signed [InBits-1:0] data_q_in = data_i[next_rom_addr];
  end

  assign data_o = OutBits'(acc_full);

endmodule
