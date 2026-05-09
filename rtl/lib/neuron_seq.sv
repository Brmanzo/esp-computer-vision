// neuron_seq.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module neuron_seq #(
   parameter int unsigned DSPCount    = 1
  ,parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits    = 32
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels = 1
  ,parameter int unsigned OutChannels = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]    Biases  = '0

  ,localparam int unsigned ChannelCountBits = InChannels  > 1 ? $clog2(InChannels)  : 1
  ,localparam int unsigned ClassCountBits   = OutChannels > 1 ? $clog2(OutChannels) : 1
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

  /* ------------------------------------ Parameters & Constants ------------------------------------ */
  // If DSPCount > OutChannels, we just use OutChannels DSPs.
  localparam int unsigned EffectiveDSPs  = (DSPCount > OutChannels) ? OutChannels : DSPCount;
  localparam int unsigned NeuronsPerDSP  = (EffectiveDSPs == 0) ? 0 : (OutChannels / EffectiveDSPs);
  localparam int unsigned LocalClassBits = (NeuronsPerDSP <= 1) ? 1 : $clog2(NeuronsPerDSP);

  /* ------------------------------------ Internal Signals ------------------------------------ */
  logic [0:0] valid_q;
  logic [ChannelCountBits-1:0] channel_counter;
  logic [LocalClassBits-1:0]   local_class_counter;
  
  wire  [0:0] last_channel      = (channel_counter == ChannelCountBits'(InChannels - 1));
  wire  [0:0] last_local_class  = (local_class_counter == LocalClassBits'(NeuronsPerDSP - 1));

  logic [0:0] busy;
  logic signed [InChannels-1:0][InBits-1:0]    data_q;
  logic signed [OutChannels-1:0][OutBits-1:0] data_out_q;

  logic [0:0] done;

  assign valid_o = valid_q;
  assign data_o  = data_out_q;

  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  // We are ready if we aren't busy and aren't holding a result.
  // We also block during the finish transition cycle.
  assign ready_o = ~busy & ~valid_q & ~done;

  /* ------------------------------------ Control Logic ------------------------------------ */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      valid_q         <= 1'b0;
      busy          <= 1'b0;
      data_q          <=   '0;
      done        <= 1'b0;
    end else begin
      if (in_fire) begin
        data_q          <= data_i;
        busy          <= 1'b1;
      end else if (busy && last_channel && last_local_class) begin
        busy          <= 1'b0;
        done        <= 1'b1;
      end
      if (done) begin
        done        <= 1'b0;
        valid_q         <= 1'b1;
      end
      if (out_fire) valid_q <= 1'b0;
    end
  end

  /* ------------------------------------ Counters ------------------------------------ */
  generate
    if (EffectiveDSPs > 0) begin : gen_dsp_counters
      counter_roll #(
         .CountBits (ChannelCountBits)
        ,.ResetVal   (0)
        ,.MaxVal     (InChannels - 1)
        ,.EnableDown (1'b0)
      ) channel_counter_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i | in_fire)
        ,.up_i       (busy)
        ,.down_i     (1'b0)
        ,.count_o    (channel_counter)
      );

      counter_roll #(
         .CountBits (LocalClassBits)
        ,.ResetVal   (0)
        ,.MaxVal     (NeuronsPerDSP - 1)
        ,.EnableDown (1'b0)
      ) class_counter_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i | in_fire)
        ,.up_i       (busy && last_channel)
        ,.down_i     (1'b0)
        ,.count_o    (local_class_counter)
      );
    end
  endgenerate

  /* ------------------------------------ Neural Logic ------------------------------------ */
  wire signed [EffectiveDSPs-1:0][OutBits-1:0] neuron_results;

  always_ff @(posedge clk_i) begin
    if (dsp_valid_q) begin
      for (int i = 0; i < EffectiveDSPs; i++) begin
        data_out_q[i * NeuronsPerDSP + int'(workload_idx)] <= neuron_results[i];
      end
    end
  end

  generate
    // Delayed capture signals for synchronous DSP outputs
    logic dsp_valid_q;
    logic [LocalClassBits-1:0] workload_idx;

    always_ff @(posedge clk_i) begin
      if (rst_i) begin
        dsp_valid_q  <= 1'b0;
        workload_idx <= '0;
      end else begin
        dsp_valid_q  <= (busy && last_channel);
        workload_idx <= local_class_counter;
      end
    end

    for (genvar dsp_idx = 0; dsp_idx < EffectiveDSPs; dsp_idx++) begin : gen_dsps
      wire [ClassCountBits-1:0] current_class = ClassCountBits'(ClassCountBits'(dsp_idx * NeuronsPerDSP) + ClassCountBits'(local_class_counter));

      wire first_channel = (channel_counter == '0);
      wire signed [WeightBits-1:0] current_weight = $signed(Weights[(current_class*WeightIndex) + (channel_counter*WeightBits) +: WeightBits]);
      wire signed [BiasBits-1:0]   current_bias   = $signed(Biases[current_class*BiasBits +: BiasBits]);
      wire signed [OutBits-1:0]    neuron_out;

      neuron_dsp #(
         .InBits     (InBits)
        ,.WeightBits (WeightBits)
        ,.OutBits    (OutBits)
        ,.BiasBits   (BiasBits)
      ) dsp_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.en_i       (busy)
        ,.load_bias_i(first_channel)
        ,.data_i     (data_q[channel_counter])
        ,.weight_i   (current_weight)
        ,.bias_i     (current_bias)
        ,.acc_o      (neuron_out)
      );

      assign neuron_results[dsp_idx] = neuron_out;
    end
  endgenerate

endmodule
