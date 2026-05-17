// neuron_seq.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module neuron_seq #(
   parameter int unsigned DSPCount    = 1
  ,parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 32
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = 1
    ,parameter FileName     = "model/data/roms/hex/zeros.hex"
  ,parameter FileName_hi  = "model/data/roms/hex/zeros.hex"
  ,parameter logic signed [OutChannels*InChannels*WeightBits-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]              Biases  = '0

  ,localparam int unsigned InChannelCountBits  = InChannels  > 1 ? $clog2(InChannels)  : 1
  ,localparam int unsigned OutChannelCountBits = OutChannels > 1 ? $clog2(OutChannels) : 1
    // If DSPCount > OutChannels, we just use OutChannels DSPs.
  ,localparam int unsigned EffectiveDSPs     = (DSPCount > OutChannels) ? OutChannels : DSPCount
  ,localparam int unsigned OutChannelsPerDSP = (EffectiveDSPs == 0) ? 0 : ((OutChannels + EffectiveDSPs - 1) / EffectiveDSPs)
  ,localparam int unsigned DSPBits           = (OutChannelsPerDSP <= 1) ? 1 : $clog2(OutChannelsPerDSP)

  ,localparam int unsigned ROMWidth    = WeightBits * EffectiveDSPs
  ,localparam int unsigned ROMDepth    = InChannels * OutChannelsPerDSP
  ,localparam int unsigned ROMAddrBits = (ROMDepth > 1) ? $clog2(ROMDepth) : 1
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

  /* -------------------------------------- Counter Logic -------------------------------------- */
  logic [InChannelCountBits-1:0]  in_ch_counter;
  logic [OutChannelCountBits-1:0] out_ch_counter;

  wire  [0:0] last_in_ch;
  wire  [0:0] last_out_ch;

  generate
    if (EffectiveDSPs > 0) begin : gen_dsp_counters
      /* verilator lint_off PINCONNECTEMPTY */
      counter_roll #(
         .CountBits  (InChannelCountBits)
        ,.ResetVal   (0)
        ,.MaxVal     (InChannels - 1)
        ,.EnableDown (1'b0)
      ) in_ch_counter_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i | in_fire)
        ,.up_i       (busy)
        ,.down_i     (1'b0)
        ,.count_o    (in_ch_counter)
        ,.next_o     ()
        ,.max_o      (last_in_ch)
      );

      counter_roll #(
         .CountBits  (OutChannelCountBits)
        ,.ResetVal   (0)
        ,.MaxVal     (OutChannelsPerDSP - 1)
        ,.EnableDown (1'b0)
      ) out_ch_counter_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i | in_fire)
        ,.up_i       (busy && last_in_ch)
        ,.down_i     (1'b0)
        ,.count_o    (out_ch_counter)
        ,.next_o     ()
        ,.max_o      (last_out_ch)
      );
      /* verilator lint_on PINCONNECTEMPTY */
    end
  endgenerate
  /* ------------------------------------ Internal Signals ------------------------------------ */
  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;

  logic [0:0] busy;

  logic signed [InChannels-1:0][InBits-1:0]   data_q;
  logic signed [OutChannels-1:0][OutBits-1:0] neuron_q;
  assign data_o = neuron_q;

  /* ------------------------------------ Control FSM ------------------------------------ */
  typedef enum logic [2:0] {Idle, Busy, Flush1, Flush2, Flush3, Done} fsm_e;
  fsm_e state_q, state_d;

  assign valid_o = (state_q == Done);
  assign ready_o = (state_q == Idle);

  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q <= Idle;
    end else begin
      state_q <= state_d;
    end
  end

  // Capture input data on in_fire and hold stable throughout DSP computation.
  // This prevents upstream changes (e.g. global_max processing a new frame)
  // from corrupting the data while DSPs are still computing.
  always_ff @(posedge clk_i) begin
    if (in_fire) data_q <= data_i;
  end

  // Next state logic
  always_comb begin
    state_d = state_q;
    busy    = (state_q == Busy);
    case (state_q)
      Idle:   if (in_fire) state_d = Busy;
      Busy:   if (last_in_ch && last_out_ch) state_d = Flush1;
      Flush1: state_d = Flush2; // en_q is still 1 (registered from last Busy); last MAC term presented
      Flush2: state_d = Flush3; // acc_r updates with last term at this edge; dsp_valid_q fires to capture neuron_q
      Flush3: state_d = Done;   // neuron_q stable; valid_o asserted next cycle
      Done:   if (out_fire) state_d = Idle;
      default: state_d = Idle;
    endcase
  end

  /* ------------------------------------ Weight ROM ------------------------------------ */
  // SB_RAM40_4K is max 16-bit wide; split wider ROMs into lo/hi tiles.
  // lo tile is always 16 bits (or ROMWidth if narrower).
  // hi tile is exactly ROMWidth - TILE_WIDTH so the concatenation {hi, lo} is ROMWidth bits.
  localparam int unsigned TILE_WIDTH = (ROMWidth > 16) ? 16 : ROMWidth;
  localparam int unsigned HI_WIDTH   = (ROMWidth > 16) ? (ROMWidth - TILE_WIDTH) : 0;

  wire [ROMWidth-1:0]    rom_weights;
  wire [ROMAddrBits-1:0] rom_addr = ROMAddrBits'(int'(out_ch_counter) * InChannels + int'(in_ch_counter));

  wire [TILE_WIDTH-1:0] rom_weights_lo;

  icestorm_rom #(
     .Width    (TILE_WIDTH)
    ,.Depth    (ROMDepth)
    ,.FileName (FileName)
  ) weight_rom_inst (
     .clk_i      (clk_i)
    ,.rst_i      (rst_i)
    ,.rd_addr_i  (rom_addr)
    ,.rd_data_o  (rom_weights_lo)
    ,.wr_valid_i (1'b0)
    ,.wr_data_i  ('0)
    ,.wr_addr_i  ('0)
  );

  generate
    if (ROMWidth > 16) begin : gen_rom_hi
      wire [HI_WIDTH-1:0] rom_weights_hi;
      icestorm_rom #(
         .Width    (HI_WIDTH)
        ,.Depth    (ROMDepth)
        ,.FileName (FileName_hi)
      ) weight_rom_hi_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.rd_addr_i  (rom_addr)
        ,.rd_data_o  (rom_weights_hi)
        ,.wr_valid_i (1'b0)
        ,.wr_data_i  ('0)
        ,.wr_addr_i  ('0)
      );
      assign rom_weights = {rom_weights_hi, rom_weights_lo};
    end else begin : gen_rom_single
      assign rom_weights = rom_weights_lo;
    end
  endgenerate

  /* ------------------------------------ DSP Capture Logic ------------------------------------ */
  wire signed [EffectiveDSPs-1:0][OutBits-1:0] neuron_results;

  always_ff @(posedge clk_i) begin
    if (dsp_valid_q) begin
      for (int i = 0; i < EffectiveDSPs; i++) begin
        neuron_q[i * OutChannelsPerDSP + int'(workload_idx)] <= neuron_results[i];
      end
    end
  end

  /* --------------------------------- DSP Instantiation --------------------------------- */
  generate
    // Delayed capture signals for synchronous DSP outputs
    // 2-stage pipeline: d1 fires when last_in_ch seen (the overlap cycle accumulates the
    // last term at this same edge); dsp_valid_q fires one edge later when acc_r is stable.
    logic dsp_valid_d1, dsp_valid_q;
    logic [OutChannelCountBits-1:0] workload_idx_d1, workload_idx;

    always_ff @(posedge clk_i) begin
      if (rst_i) begin
        dsp_valid_d1    <= 1'b0;
        workload_idx_d1 <= '0;

        dsp_valid_q  <= 1'b0;
        workload_idx <= '0;
      end else begin
        dsp_valid_d1    <= (busy && last_in_ch);
        workload_idx_d1 <= out_ch_counter;

        dsp_valid_q  <= dsp_valid_d1;
        workload_idx <= workload_idx_d1;
      end
    end

    for (genvar dsp_idx = 0; dsp_idx < EffectiveDSPs; dsp_idx++) begin : gen_dsps
      wire [OutChannelCountBits-1:0] current_class = OutChannelCountBits'(OutChannelCountBits'(dsp_idx * OutChannelsPerDSP) + OutChannelCountBits'(out_ch_counter));

      wire [0:0] first_channel = (in_ch_counter == '0);
      wire signed [WeightBits-1:0] current_weight = $signed(rom_weights[dsp_idx*WeightBits +: WeightBits]);
      wire signed [BiasBits-1:0]   current_bias   = $signed(Biases[current_class*BiasBits +: BiasBits]);
      
      wire signed [OutBits-1:0]        neuron_out;
      assign neuron_results[dsp_idx] = neuron_out;

      logic [0:0] en_q;
      logic [0:0] load_bias_q;

      // InBits is already InBits+1 (zero-extended) from linear_layer's input_encoder.
      // No additional extension needed here.
      logic signed [InBits-1:0] neuron_in;
      logic signed [BiasBits-1:0] bias_q;
      always_ff @(posedge clk_i) begin
        // Enable SB_MAC16 when busy, 1 cycle delay to align with ROM
        en_q <= busy;
        // Capture current input off of input
        neuron_in <= data_q[in_ch_counter];
        // Capture bias and load on first term
        load_bias_q <= first_channel;
        bias_q      <= current_bias;
      end

      neuron_dsp #(
         .InBits     (InBits)
        ,.WeightBits (WeightBits)
        ,.OutBits    (OutBits)
        ,.BiasBits   (BiasBits)
      ) dsp_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.en_i       (en_q)
        ,.load_bias_i(load_bias_q)
        ,.data_i     (neuron_in)
        ,.weight_i   (current_weight)           // Delayed 1 cycle by icestorm_rom
        ,.bias_i     (bias_q)
        ,.acc_o      (neuron_out)
      );

    end
  endgenerate

endmodule
