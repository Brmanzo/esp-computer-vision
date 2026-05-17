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
  ,parameter int unsigned Unsigned    = (InBits > 2) ? 1:0
  ,parameter int unsigned ShiftBits   = 0

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]    Biases  = '0
  ,parameter int unsigned DSPCount  = 0 // 0: LUT, 1: Sequential DSP per class, 2: Fully Sequential DSP (one total)
  ,parameter FileName    = "model/data/roms/hex/zeros.hex"
  ,parameter FileName_hi = "model/data/roms/hex/zeros.hex"

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

  logic [0:0] valid_q;
  logic signed [InChannels-1:0][(InBits):0] data_q;  // InBits+1 wide: zero-extended for unsigned activations
  logic signed [OutChannels-1:0][OutBits-1:0] data_out_q;

  // Zero-extend each input channel from InBits to InBits+1 (unsigned post-ReLU contract).
  // Each channel gets a 0 prepended as the sign bit so downstream signed multipliers
  // treat the full InBits as magnitude, giving range [0, 2^InBits - 1].
  wire signed [InChannels-1:0][(InBits):0] encoded_i;
  generate
    for (genvar enc_ch = 0; enc_ch < InChannels; enc_ch++) begin : gen_encode
      input_encoder #(
         .Unsigned(Unsigned)
        ,.InBits  (InBits)
        ,.OutBits (InBits + 1)
      ) input_encoder_inst (
         .data_i(data_i[enc_ch])
        ,.data_o(encoded_i[enc_ch])
      );
    end
  endgenerate

  generate
    if (DSPCount > 0) begin : gen_seq_neurons
      neuron_seq #(
        .DSPCount     (DSPCount)
        ,.InBits      (InBits + 1)
        ,.OutBits     (OutBits)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.InChannels  (InChannels)
        ,.OutChannels (OutChannels)
        ,.Weights     (Weights)
        ,.Biases      (Biases)
        ,.FileName    (FileName)
        ,.FileName_hi (FileName_hi)
      ) filter_select_inst (
        .clk_i   (clk_i)
        ,.rst_i   (rst_i)
        ,.ready_o (ready_o)
        ,.valid_i (valid_i)
        ,.data_i  (encoded_i)
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
            data_q  <= encoded_i;
          end
        end
      end

      for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
        neuron #(
           .InBits    (InBits + 1)
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
        ,.ShiftBits(ShiftBits)
      ) out_enc_inst (
         .data_i(data_out_q[ch])
        ,.data_o(data_o[ch])
      );
    end
  endgenerate

endmodule
