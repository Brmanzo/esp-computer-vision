// filter_seq.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module filter_seq #(
   parameter int unsigned DSPCount    = 1
  ,parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = 8
  ,parameter int unsigned InChannels = 1
  ,parameter int unsigned OutChannels = 1
  ,parameter int unsigned KernelWidth = 3
  ,parameter int unsigned AccBits     = 32

  ,localparam int unsigned KernelArea = KernelWidth * KernelWidth
  ,localparam int unsigned WeightIndex = InChannels * KernelArea * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]    Biases  = '0

  ,localparam int unsigned ClassCountBits = (OutChannels > 1) ? $clog2(OutChannels) : 1
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,output [0:0] ready_o
  ,input  [0:0] valid_i
  ,input  logic signed [InChannels*KernelArea*InBits-1:0] windows_i

  ,input  [0:0] ready_i
  ,output [0:0] valid_o
  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

  /* ------------------------------------ Parameters & Constants ------------------------------------ */
  localparam int unsigned EffectiveDSPs  = (DSPCount > OutChannels) ? OutChannels : DSPCount;
  localparam int unsigned NeuronsPerDSP  = (EffectiveDSPs == 0) ? 0 : ((OutChannels + EffectiveDSPs - 1) / EffectiveDSPs);
  localparam int unsigned LocalClassBits = (NeuronsPerDSP <= 1) ? 1 : $clog2(NeuronsPerDSP);

  /* ------------------------------------ Internal Signals ------------------------------------ */
  logic [0:0] valid_q;
  logic [LocalClassBits-1:0] local_class_counter;
  
  wire  [0:0] last_local_class = (local_class_counter == LocalClassBits'(NeuronsPerDSP - 1));

  logic [0:0] busy;
  logic signed [InChannels*KernelArea*InBits-1:0] data_q;
  logic signed [OutChannels-1:0][OutBits-1:0] data_out_q;

  logic [0:0] done;

  assign valid_o = valid_q;
  assign data_o  = data_out_q;

  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;

  assign ready_o = ~busy & ~valid_q & ~done;

  /* ------------------------------------ Control Logic ------------------------------------ */
  wire [EffectiveDSPs-1:0] dsp_done;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      valid_q   <= 1'b0;
      busy      <= 1'b0;
      data_q    <= '0;
      done      <= 1'b0;
      local_class_counter <= '0;
    end else begin
      if (in_fire) begin
        data_q <= windows_i;
        busy   <= 1'b1;
        local_class_counter <= '0;
      end else if (busy && dsp_done[0]) begin
        if (last_local_class) begin
          busy <= 1'b0;
          done <= 1'b1;
        end else begin
          local_class_counter <= local_class_counter + 1;
        end
      end

      if (done) begin
        done    <= 1'b0;
        valid_q <= 1'b1;
      end

      if (out_fire) valid_q <= 1'b0;
    end
  end

  /* ------------------------------------ Neural Logic ------------------------------------ */
  generate
    // Capture signal for results
    wire dsp_valid_w = (busy && dsp_done[0]);

    // Combinatorial class selection logic to avoid off-by-one during in_fire
    wire [LocalClassBits-1:0] next_class_counter = in_fire ? '0 : 
                                                   (busy && dsp_done[0] && !last_local_class) ? (local_class_counter + 1) : 
                                                   local_class_counter;

    for (genvar dsp_idx = 0; dsp_idx < EffectiveDSPs; dsp_idx++) begin : gen_dsps
      wire [ClassCountBits-1:0] next_class    = ClassCountBits'(ClassCountBits'(dsp_idx * NeuronsPerDSP) + next_class_counter);
      wire [ClassCountBits-1:0] capture_class = ClassCountBits'(ClassCountBits'(dsp_idx * NeuronsPerDSP) + local_class_counter);

      wire signed [WeightIndex-1:0] current_weight = Weights[next_class*WeightIndex +: WeightIndex];
      wire signed [BiasBits-1:0]    current_bias   = Biases[next_class*BiasBits +: BiasBits];
      wire signed [OutBits-1:0]     filter_out;

      // We start class 0 on in_fire, subsequent classes on dsp_done of previous class
      logic [0:0] dsp_start_pulse;
      logic [0:0] dummy_ready_o;
      always_ff @(posedge clk_i) begin
        if (rst_i) dsp_start_pulse <= 1'b0;
        else       dsp_start_pulse <= (busy && dsp_done[dsp_idx] && !last_local_class);
      end

      filter_dsp #(
         .InBits      (InBits)
        ,.OutBits     (OutBits)
        ,.KernelWidth (KernelWidth)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.AccBits     (AccBits)
        ,.InChannels  (InChannels)
        ,.DSPIdx      (dsp_idx)
      ) dsp_inst (
         .clk_i    (clk_i)
        ,.rst_i    (rst_i)
        ,.valid_i  (in_fire | dsp_start_pulse)
        ,.ready_o  (dummy_ready_o)
        ,.windows_i(in_fire ? windows_i : data_q)
        ,.weights_i(current_weight)
        ,.bias_i   (current_bias)
        ,.ready_i  (1'b1) // Sequential capture, always ready for result
        ,.valid_o  (dsp_done[dsp_idx])
        ,.data_o   (filter_out)
      );

      always_ff @(posedge clk_i) begin
        if (dsp_valid_w) begin
          if (32'(capture_class) < OutChannels) begin
            data_out_q[capture_class] <= filter_out;
          end
        end
      end
    end
  endgenerate

endmodule
