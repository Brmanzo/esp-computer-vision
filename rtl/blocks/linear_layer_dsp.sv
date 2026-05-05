// linear_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module linear_layer_dsp #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter  logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter  logic signed [OutChannels*BiasBits-1:0]    Biases  = '0
  ,localparam int unsigned ChannelCountBits = InChannels > 1 ? $clog2(InChannels) : 1
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

  /* ------------------------------------ Sequential Control Logic ------------------------------------ */
  logic [0:0] valid_r;
  logic [ChannelCountBits-1:0] channel_counter;
  wire  first_channel = (channel_counter == '0);
  wire  last_channel  = (channel_counter == ChannelCountBits'(InChannels - 1));

  logic [0:0] busy_q;
  logic signed [InChannels-1:0][InBits-1:0] data_q;

  // We are ready for a new frame if we aren't currently busy and aren't holding a valid result
  assign ready_o = ~busy_q & ~valid_r;
  assign valid_o =  valid_r;

  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      channel_counter <= '0;
      valid_r         <= 1'b0;
      busy_q          <= 1'b0;
      data_q          <= '0;
    end else begin
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
    end
  end

  /* --------------------------------------- Output Channel Logic --------------------------------------- */
  
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
      // Multiplex the weight for the current channel
      wire signed [WeightBits-1:0] current_weight = Weights[(ch*WeightIndex) + (channel_counter*WeightBits) +: WeightBits];
      
      neuron_dsp #(
         .InBits    (InBits)
        ,.WeightBits(WeightBits)
        ,.AccBits   (OutBits)
      ) neuron_dsp_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.en_i       (in_fire | busy_q) // Active during capture AND accumulation
        ,.load_bias_i(first_channel)    // Load bias only on the first cycle
        ,.data_i     (in_fire ? data_i[0] : data_q[channel_counter])
        ,.weight_i   (current_weight)
        ,.bias_i     (OutBits'($signed(Biases[(ch*BiasBits) +: BiasBits]))) // Truncate/Extend to AccBits
        ,.acc_o      (data_o[ch])
      );
    end
  endgenerate

endmodule
