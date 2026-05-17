// filter_seq.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

module filter_seq #(
   parameter int unsigned DSPCount    = 1
  ,parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = 1
  ,parameter int unsigned ShiftBits   = 0
  ,parameter int unsigned KernelWidth = 3
  ,parameter int unsigned AccBits     = 32
  ,parameter int unsigned Unsigned    = 0

  ,localparam int unsigned KernelArea = KernelWidth * KernelWidth
  ,parameter FileName    = "model/data/roms/hex/zeros.hex"
  // second 16-bit tile for ROMs wider than one SB_RAM40_4K
  ,parameter FileName_hi = "model/data/roms/hex/zeros.hex"
  // Conditional loading of Weights from ROM or parameter
  ,parameter logic signed [OutChannels*BiasBits-1:0]                         Biases  = '0
  ,localparam int unsigned ChannelCountBits = (OutChannels > 1) ? $clog2(OutChannels) : 1
  // DSP Indexing 
  ,localparam int unsigned EffectiveDSPs     = (DSPCount > OutChannels) ? OutChannels : DSPCount
  ,localparam int unsigned OutChannelsPerDSP = (EffectiveDSPs == 0) ? 0 : ((OutChannels + EffectiveDSPs - 1) / EffectiveDSPs)
  ,localparam int unsigned DSPBits           = (OutChannelsPerDSP <= 1) ? 1 : $clog2(OutChannelsPerDSP)
  // ROM Dimensioning
  ,localparam int unsigned ROMWidth    = WeightBits * EffectiveDSPs
  ,localparam int unsigned TotalTerms  = InChannels * KernelArea
  ,localparam int unsigned TermBits    = (TotalTerms > 1) ? $clog2(TotalTerms) : 1
  ,localparam int unsigned ROMDepth    = TotalTerms * OutChannelsPerDSP
  ,localparam int unsigned ROMAddrBits = (ROMDepth > 1) ? $clog2(ROMDepth) : 1
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

  /* ------------------------------------ Internal Signals ------------------------------------ */
  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;

  // DSPs advance in parallel, sample DSP_0 to determine if valid output
  wire  [EffectiveDSPs-1:0] dsp_done;
  logic [0:0] busy;
  wire  [0:0] dsps_valid = (busy && dsp_done[0]);
  /* ------------------------------------ Counter Logic ------------------------------------ */
  logic [DSPBits-1:0] channel_count_q;
  wire  [0:0]         out_channel_done;

  wire [DSPBits-1:0] channel_count_d;

  // Tracks completion of outchannel workload on DSPs
  /* verilator lint_off PINCONNECTEMPTY */
  counter_roll #(
     .CountBits  (DSPBits)
    ,.ResetVal   (0)
    ,.MaxVal     (OutChannelsPerDSP - 1)
    ,.EnableDown (1'b0)
  ) out_channel_counter_inst (
     .clk_i      (clk_i)
    ,.rst_i      (rst_i | in_fire)
    ,.up_i       (dsps_valid)
    ,.down_i     (1'b0)
    ,.count_o    (channel_count_q)
    ,.next_o     (channel_count_d)
    ,.max_o      (out_channel_done)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  /* ------------------------------------ Control FSM ------------------------------------ */
  typedef enum logic [1:0] {Idle, Busy, Done} fsm_e;
  fsm_e state_q, state_d;

  // Capture filter input on in-fire
  logic signed [InChannels*KernelArea*InBits-1:0] windows_q;
  
  // Valid Data when Done
  assign valid_o = (state_q == Done);
  // Ready to consume when Idle
  assign ready_o = (state_q == Idle);

  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q   <= Idle;
      windows_q <= '0;
    end else begin
      state_q   <= state_d;
      // Register window input for DSP
      if (in_fire) windows_q <= windows_i;
    end
  end

  // Next state logic
  always_comb begin
    state_d = state_q;
    busy    = (state_q == Busy);
    case (state_q)
      Idle: if (in_fire)  state_d = Busy;
      Busy: if (dsp_done[0] && out_channel_done) state_d = Done;
      Done: if (out_fire) state_d = Idle;
      default: state_d = Idle;
    endcase
  end

  /* ------------------------------------- ROM Dimensioning ------------------------------------- */
  // SB_RAM40_4K is max 16-bit wide; split wider ROMs into lo/hi tiles.
  // lo tile is always 16 bits (or ROMWidth if narrower).
  // hi tile is exactly ROMWidth - TILE_WIDTH so {hi, lo} is ROMWidth bits.
  localparam int unsigned TILE_WIDTH = (ROMWidth > 16) ? 16 : ROMWidth;
  localparam int unsigned HI_WIDTH   = (ROMWidth > 16) ? (ROMWidth - TILE_WIDTH) : 0;

  wire [ROMWidth-1:0]    rom_weights;
  wire [ROMAddrBits-1:0] rom_addr; // Driven by DSP term_idx and channel_count_q

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
  logic signed [OutChannels-1:0][OutBits-1:0] filter_o;
  assign data_o = filter_o;
  
  wire signed  [EffectiveDSPs-1:0][OutBits-1:0] dsp_o;

  // Pack DSP outputs onto filter output
  always_ff @(posedge clk_i) begin
    if (dsps_valid) begin
      for (int dsp = 0; dsp < EffectiveDSPs; dsp++) begin
        if (dsp * OutChannelsPerDSP + int'(channel_count_q) < int'(OutChannels)) begin
          filter_o[dsp * OutChannelsPerDSP + int'(channel_count_q)] <= dsp_o[dsp];
        end
      end
    end
  end

  /* ------------------------------------ DSP Instantiation ------------------------------------ */
  generate
    // All DSPs advance in parallel sharing the same counter
    assign rom_addr = ROMAddrBits'(int'(channel_count_d) * TotalTerms + int'(gen_dsps[0].next_dsp_term_idx));

    for (genvar dsp = 0; dsp < EffectiveDSPs; dsp++) begin : gen_dsps
      wire [ChannelCountBits-1:0] next_class   = ChannelCountBits'(ChannelCountBits'(dsp * OutChannelsPerDSP) + channel_count_d);
      wire [TermBits-1:0]         dsp_term_idx;
      
      // We start class 0 on in_fire, subsequent classes on dsp_done of previous class
      logic [0:0] dsp_start_pulse;
      always_ff @(posedge clk_i) begin
        if (rst_i) dsp_start_pulse <= 1'b0;
        else       dsp_start_pulse <= (busy && dsp_done[dsp] && !out_channel_done);
      end

      wire [TermBits-1:0]          next_dsp_term_idx = dsp_term_idx;

      // Index weight from ROM, and bias off of parameter
      wire signed [WeightBits-1:0] dsp_weight = $signed(rom_weights[dsp*WeightBits +: WeightBits]);
      wire signed [BiasBits-1:0]   dsp_bias   = Biases[next_class*BiasBits +: BiasBits];

      /* verilator lint_off PINCONNECTEMPTY */
      filter_dsp #(
         .InBits      (InBits)
        ,.OutBits     (OutBits)
        ,.KernelWidth (KernelWidth)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.AccBits     (AccBits)
        ,.InChannels  (InChannels)
        ,.ShiftBits   (ShiftBits)
        ,.Unsigned    (Unsigned)
        ,.DSPIdx      (dsp)
      ) dsp_inst (
         .clk_i (clk_i)
        ,.rst_i (rst_i)

        ,.valid_i   (in_fire | dsp_start_pulse)
        ,.ready_o   ()
        ,.windows_i (in_fire ? windows_i : windows_q)
        ,.weight_i  (dsp_weight)
        ,.bias_i    (dsp_bias)

        ,.ready_i    (1'b1)
        ,.valid_o    (dsp_done[dsp])
        ,.term_idx_o (dsp_term_idx)
        ,.data_o     (dsp_o[dsp])
      );
      /* verilator lint_on PINCONNECTEMPTY */
    end
  endgenerate

endmodule
