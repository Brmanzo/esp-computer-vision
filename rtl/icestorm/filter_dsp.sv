// filter_dsp.sv
// Bradley Manzo 2026

// Targetting IceStorm's 8 SB_MAC16 DSPs
// https://0x04.net/~mwk/sbdocs/ice40/FPGA-TN-02007-1-2-DSP-Function-Usage-Guide-for-iCE40-Devices.pdf

`define SB_MAC16_IN  16
`define SB_MAC16_OUT 32

`timescale 1ns/1ps
module filter_dsp #(
   parameter   int unsigned InBits      = 1
  ,parameter   int unsigned OutBits     = 1
  ,parameter   int unsigned KernelWidth = 3
  ,parameter   int unsigned WeightBits  = 2
  ,parameter   int unsigned BiasBits    = 8
  ,parameter   int unsigned AccBits     = `SB_MAC16_OUT
  ,parameter   int unsigned InChannels  = 4
  ,parameter   int unsigned DSPIdx      = 0

  ,localparam  int unsigned KernelArea  = KernelWidth * KernelWidth
  ,localparam  int unsigned TotalTerms  = InChannels * KernelArea
  ,localparam  int unsigned TermBits    = (TotalTerms > 1) ? $clog2(TotalTerms) : 1
) (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o

  ,input logic signed [InChannels*KernelArea*InBits-1:0] windows_i
  ,output [TermBits-1:0]                                 term_idx_o
  ,input logic signed [WeightBits-1:0]                   weight_i
  ,input logic signed [BiasBits-1:0]                     bias_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutBits-1:0] data_o
);

  /* ------------------------ Internal Signals ------------------------ */
  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;
    /* ---------------------------- Counter Logic ---------------------------- */
  logic [TermBits-1:0] term_count_q;

  wire  [0:0] new_term = (term_count_q == '0);
  wire  [0:0] last_term;
  logic [0:0] busy;

  /* verilator lint_off PINCONNECTEMPTY */
  counter_roll #(
      .CountBits  (TermBits)
    ,.ResetVal   (0)
    ,.MaxVal     (TotalTerms - 1)
    ,.EnableDown (1'b0)
  ) term_counter_inst (
      .clk_i      (clk_i)
    ,.rst_i      (rst_i | in_fire)
    ,.up_i       (busy)
    ,.down_i     (1'b0)
    ,.count_o    (term_count_q)
    ,.next_o     ()
    ,.max_o      (last_term)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  /* ----------------------------- FSM Logic ----------------------------- */
  typedef enum logic [1:0] {Idle, Busy, Flush, Done} fsm_e;
  fsm_e state_q, state_d;
  assign valid_o = (state_q == Done);
  assign ready_o = (state_q == Idle);

  // Current State Logic
  always_ff @(posedge clk_i) begin
    if(rst_i) state_q <= Idle;
    else      state_q <= state_d;
  end

  // Next State Logic
  always_comb begin
    state_d = state_q;
    busy    = (state_q == Busy);
    case (state_q)
      Idle:  if (in_fire)   state_d = Busy;
      Busy:  if (last_term) state_d = Flush;
      Flush:                state_d = Done; // Wait 1 cycle for pipeline
      Done:  if (out_fire)  state_d = Idle;
      default:              state_d = Idle;
    endcase
  end

  /* ------------------------ Data Sequencing ------------------------ */
  // term_idx follows term_count_q during busy cycles
  wire [TermBits-1:0] term_idx = term_count_q;
  assign term_idx_o = term_idx;

  // Access weights and windows directly without registration
  logic [0:0] en_q;
  logic [0:0] load_bias_r;
  logic signed [InBits-1:0]   data_q;
  logic signed [BiasBits-1:0] bias_q;

  always_ff @(posedge clk_i) begin
    // Enable SB_MAC16 when busy, 1 cycle delay to align with ROM
    en_q   <= busy;
    // Capture current input off of window
    data_q <= windows_i[term_idx*InBits +: InBits];
    // Capture bias and load on first term
    bias_q      <= bias_i;
    load_bias_r <= (busy && new_term);
  end

  /* ------------------------------- Neuron DSP Unit ------------------------------- */
  // Targetting IceStorm SB_MAC16 DSP
  // Acts as sequencer, allows consumer DSP to handle encoding
  wire signed [`SB_MAC16_OUT-1:0] neuron_o;

  neuron_dsp #(
     .InBits     (InBits)
    ,.WeightBits (WeightBits)
    ,.BiasBits   (BiasBits)
    ,.OutBits    (`SB_MAC16_OUT)
  ) dsp_inst (
     .clk_i       (clk_i)
    ,.rst_i       (rst_i)
    ,.en_i        (en_q)
    ,.load_bias_i (load_bias_r) // Load Bias only on first cycle
    ,.data_i      (data_q)
    ,.weight_i    (weight_i) // 1-cycle delayed from ROM
    ,.bias_i      (bias_q)
    ,.acc_o       (neuron_o)
    );
    /* -------------------------------- Output Encoding -------------------------------- */
    output_encoder #(
       .InBits  (AccBits)
      ,.OutBits (OutBits)
    ) out_enc_inst (
       .data_i (neuron_o[AccBits-1:0])
      ,.data_o (data_o)
    );
  endmodule
