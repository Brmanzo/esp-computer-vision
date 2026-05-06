// linear_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module linear_layer #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels   = 1
  ,parameter int unsigned OutChannels  = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]   Biases  = '0
  ,parameter int unsigned UseDSP  = 0 // Set to 1 to use sequential DSP implementation
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

  localparam int unsigned ChannelCountBits = InChannels > 1 ? $clog2(InChannels) : 1;

  /* ------------------------------------ Sequential Control Logic ------------------------------------ */
  logic [0:0] valid_r;
  logic [ChannelCountBits-1:0] channel_counter;
  wire  last_channel  = (channel_counter == ChannelCountBits'(InChannels - 1));

  logic [0:0] busy_q;
  logic signed [InChannels-1:0][InBits-1:0] data_q;

  // We are ready for a new frame if we aren't currently busy and aren't holding a valid result
  assign ready_o = (UseDSP != 0) ? (~busy_q & ~valid_r) : (~valid_r | ready_i);
  assign valid_o = valid_r;

  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      channel_counter <= '0;
      valid_r         <= 1'b0;
      busy_q          <= 1'b0;
      data_q          <= '0;
    end else if (UseDSP != 0) begin
      // 1. Capture and Start Processing
      if (in_fire) begin
        data_q          <= data_i;
        busy_q          <= 1'b1;
        channel_counter <= '0;
      end 
      // 2. Sequential Accumulation
      else if (busy_q) begin
        if (last_channel) begin
          busy_q          <= 1'b0;
          valid_r         <= 1'b1;
        end else begin
          channel_counter <= channel_counter + 1;
        end
      end
      // 3. Output Handshake
      if (out_fire) begin
        valid_r <= 1'b0;
      end
    end else begin
      // Original Combinational/LUT logic
      if (ready_o) begin
        valid_r <= in_fire;
        data_q  <= data_i;
      end
    end
  end

  /* --------------------------------------- Output Channel Logic --------------------------------------- */
  
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
      if (UseDSP == 1) begin : gen_neuron_dsp
        // Multiplex the weight for the current channel
        wire first_channel = (channel_counter == '0);

        wire signed [WeightBits-1:0] current_weight = Weights[(ch*WeightIndex) + (channel_counter*WeightBits) +: WeightBits];
        wire signed [BiasBits-1:0]   current_bias   = Biases[ch*BiasBits +: BiasBits];
        
        neuron_dsp #(
           .InBits     (InBits)
          ,.WeightBits (WeightBits)
          ,.OutBits    (OutBits)
          ,.BiasBits   (BiasBits)
        ) neuron_dsp_inst (
           .clk_i      (clk_i)
          ,.rst_i      (rst_i)
          ,.en_i       (in_fire | busy_q) 
          ,.load_bias_i(first_channel)    
          ,.data_i     (in_fire ? data_i[0] : data_q[channel_counter])
          ,.weight_i   (current_weight)
          ,.bias_i     (current_bias) 
          ,.acc_o      (data_o[ch])
        );
      end else begin : gen_neuron_lut
        neuron #(
           .InBits    (InBits)
          ,.OutBits   (OutBits)
          ,.WeightBits(WeightBits)
          ,.BiasBits  (BiasBits)
          ,.InChannels (InChannels)

          ,.Weights(Weights[ch*WeightIndex +: WeightIndex])
          ,.Bias   (Biases[ch*BiasBits +: BiasBits])
        ) neuron_inst (
           .data_i(data_q)
          ,.data_o(data_o[ch])
        );
      end
    end
  endgenerate

endmodule
