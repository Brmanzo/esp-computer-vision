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
  ,parameter int unsigned UseDSP  = 0 // 0: LUT, 1: Sequential DSP per class, 2: Fully Sequential DSP (one total)
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
  localparam int unsigned ClassCountBits   = OutChannels > 1 ? $clog2(OutChannels) : 1;

  /* ------------------------------------ Sequential Control Logic ------------------------------------ */
  logic [0:0] valid_r;
  logic [ChannelCountBits-1:0] channel_counter;
  logic [ClassCountBits-1:0]   class_counter;
  
  wire  last_channel  = (channel_counter == ChannelCountBits'(InChannels - 1));
  wire  last_class    = (class_counter   == ClassCountBits'(OutChannels - 1));

  logic [0:0] busy_q;
  logic signed [InChannels-1:0][InBits-1:0] data_q;
  logic signed [OutChannels-1:0][OutBits-1:0] data_out_q;

  // UseDSP=2 specific timing signals
  logic capture_v;
  logic [ClassCountBits-1:0] capture_idx;
  logic finish_q;

  // We are ready for a new frame if we aren't currently busy and aren't holding a valid result
  // In UseDSP=2 mode, we must also block while finish_q is high (transition cycle)
  assign ready_o = (UseDSP == 2) ? (~busy_q & ~valid_r & ~finish_q) : 
                   (UseDSP == 1) ? (~busy_q & ~valid_r) : 
                   (~valid_r | ready_i);

  assign valid_o = valid_r;
  assign data_o  = data_out_q;

  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      channel_counter <= '0;
      class_counter   <= '0;
      valid_r         <= 1'b0;
      busy_q          <= 1'b0;
      data_q          <= '0;
      capture_v       <= 1'b0;
      capture_idx     <= '0;
      finish_q        <= 1'b0;
    end else if (UseDSP == 1) begin
      // Classes parallel, channels sequential (one DSP per class)
      if (in_fire) begin
        data_q          <= data_i;
        busy_q          <= 1'b1;
        channel_counter <= '0;
      end else if (busy_q) begin
        if (last_channel) begin
          busy_q          <= 1'b0;
          valid_r         <= 1'b1;
        end else begin
          channel_counter <= channel_counter + 1;
        end
      end
      if (out_fire) valid_r <= 1'b0;
    end else if (UseDSP == 2) begin
      // Classes sequential, channels sequential (one total DSP)
      capture_v <= 1'b0; // Default
      
      if (in_fire) begin
        data_q          <= data_i;
        busy_q          <= 1'b1;
        channel_counter <= '0;
        class_counter   <= '0;
      end else if (busy_q) begin
        if (last_channel) begin
          channel_counter <= '0;
          capture_v       <= 1'b1;
          capture_idx     <= class_counter;
          
          if (last_class) begin
            busy_q   <= 1'b0;
            finish_q <= 1'b1;
          end else begin
            class_counter <= class_counter + 1;
          end
        end else begin
          channel_counter <= channel_counter + 1;
        end
      end

      if (finish_q) begin
        finish_q <= 1'b0;
        valid_r  <= 1'b1;
      end

      if (out_fire) valid_r <= 1'b0;
    end else begin
      // LUT/Parallel
      if (ready_o) begin
        valid_r <= in_fire;
        data_q  <= data_i;
      end
    end
  end

  /* --------------------------------------- Output Channel Logic --------------------------------------- */
  
  generate
    if (UseDSP == 2) begin : gen_fully_sequential_dsp
      // SINGLE neuron_dsp instance shared across ALL classes
      wire first_channel = (channel_counter == '0);
      wire signed [WeightBits-1:0] current_weight = Weights[(class_counter*WeightIndex) + (channel_counter*WeightBits) +: WeightBits];
      wire signed [BiasBits-1:0]   current_bias   = Biases[class_counter*BiasBits +: BiasBits];
      wire signed [OutBits-1:0]    neuron_out;

      neuron_dsp #(
         .InBits     (InBits)
        ,.WeightBits (WeightBits)
        ,.OutBits    (OutBits)
        ,.BiasBits   (BiasBits)
      ) neuron_dsp_single (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.en_i       (busy_q) 
        ,.load_bias_i(first_channel)    
        ,.data_i     (data_q[channel_counter])
        ,.weight_i   (current_weight)
        ,.bias_i     (current_bias) 
        ,.acc_o      (neuron_out)
      );

      // Register the output one cycle after the class finishes to capture synchronous DSP result
      always_ff @(posedge clk_i) begin
        if (capture_v) begin
          data_out_q[capture_idx] <= neuron_out;
        end
      end

    end else if (UseDSP == 1) begin : gen_parallel_class_dsp
      for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
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
          ,.acc_o      (data_out_q[ch])
        );
      end
    end else begin : gen_neuron_lut
      for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
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
          ,.data_o(data_out_q[ch])
        );
      end
    end
  endgenerate

endmodule
