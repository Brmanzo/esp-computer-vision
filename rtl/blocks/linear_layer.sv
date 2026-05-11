// linear_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module linear_layer #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]    Biases  = '0
  ,parameter int unsigned DSPCount  = 0 // 0: LUT, 1: Sequential DSP per class, 2: Fully Sequential DSP (one total)
  ,parameter string       FileName  = "memory_init_file.hex"

)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,output [0:0] ready_o
  ,input  [0:0] valid_i
  ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

  ,input  [0:0] ready_i
  ,output [0:0] valid_o
  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

  logic signed [InChannels*InBits-1:0] data_q;
  logic [0:0] valid_q;
  logic signed [OutChannels-1:0][OutBits-1:0] data_out_q;

  generate
    if (DSPCount > 0) begin : gen_seq_neurons
      neuron_seq #(
        .DSPCount    (DSPCount)
        ,.InBits      (InBits)
        ,.OutBits     (OutBits)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.InChannels  (InChannels)
        ,.OutChannels (OutChannels)
        ,.Weights     (Weights)
        ,.Biases      (Biases)
        ,.FileName    (FileName)
      ) filter_select_inst (
        .clk_i   (clk_i)
        ,.rst_i   (rst_i)
        ,.ready_o (ready_o)
        ,.valid_i (valid_i)
        ,.data_i  (data_i)
        ,.ready_i (ready_i)
        ,.valid_o (valid_o)
        ,.data_o  (data_out_q)
      );
    end else begin : gen_parallel_neurons
      wire  [0:0] in_fire  = valid_i && ready_o;

      assign valid_o = valid_q;
      assign ready_o = (~valid_q | ready_i);

      always_ff @(posedge clk_i) begin
        if (rst_i) begin
          valid_q <= 1'b0;
          data_q  <= '0;
        end else begin
          if (ready_o) begin
            valid_q <= in_fire;
            data_q  <= data_i;
          end
        end
      end

      for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
        neuron #(
           .InBits    (InBits)
          ,.OutBits   (OutBits)
          ,.WeightBits(WeightBits)
          ,.BiasBits  (BiasBits)
          ,.InChannels(InChannels)
          ,.Weights($signed(Weights[ch*WeightIndex +: WeightIndex]))
          ,.Bias   ($signed(Biases[ch*BiasBits +: BiasBits]))
        ) neuron_inst (
           .data_i(data_q)
          ,.data_o(data_out_q[ch])
        );
      end
    end
  endgenerate

  /* --------------------------------------- Output Quantization --------------------------------------- */
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_out_enc
      output_encoder #(
         .InBits (OutBits)
        ,.OutBits(OutBits)
      ) out_enc_inst (
         .data_i(data_out_q[ch])
        ,.data_o(data_o[ch])
      );
    end
  endgenerate

endmodule
