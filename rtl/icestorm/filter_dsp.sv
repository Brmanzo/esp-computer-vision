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

  ,localparam  int unsigned KernelArea  = KernelWidth * KernelWidth
  ,localparam  int unsigned TotalTerms  = InChannels * KernelArea
  ,localparam  int unsigned TermBits    = (TotalTerms > 1) ? $clog2(TotalTerms) : 1
) (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o

  ,input logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_i
  ,input logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_i
  ,input logic signed [BiasBits-1:0]   bias_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutBits-1:0] data_o
);

  /* ------------------------ Internal Signals ------------------------ */
  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;

  // Registered window and register arrays
  logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_q;
  logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_q;

  /* -------------------------- Term Counter Logic -------------------------- */
  logic [TermBits-1:0] term_count_d, term_count_q;

  wire [0:0] last_term   = (term_count_q == TermBits'(TotalTerms - 1));
  wire [0:0] single_term = (TotalTerms == 1);

  always_ff @(posedge clk_i) begin
    if (rst_i) term_count_q <= '0;
    else       term_count_q <= term_count_d;
  end

  always_comb begin
    term_count_d = term_count_q;
    if (in_fire) begin
      // If multi-term, the first term (0) is done now, so next cycle we process term 1
      term_count_d = single_term ? TermBits'(0) : TermBits'(1);
    end else if (busy_q) begin
      // Roll over to zero on last term
      if (last_term) term_count_d = '0;
      else           term_count_d = term_count_q + TermBits'(1);
    end
  end

  /* -------------------------- Elastic Handshake -------------------------- */
  logic [0:0] busy_d,  busy_q;
  logic [0:0] valid_d, valid_q;

  assign ready_o = ~busy_q & ~valid_q;
  assign valid_o = valid_q;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      busy_q    <= 1'b0;
      valid_q   <= 1'b0;
      windows_q <=   '0;
      weights_q <=   '0;
    end else begin
      busy_q       <= busy_d;
      valid_q      <= valid_d;
      if (in_fire) begin
        windows_q  <= windows_i;
        weights_q  <= weights_i;
      end
    end
  end

  always_comb begin
    busy_d  = busy_q;
    valid_d = valid_q;
    
    // When calculating single term, completes immediately
    if (single_term) begin
      if (in_fire) valid_d = 1'b1;
    // Otherwise, busy during first terms, valid after last term
    end else begin
      if (in_fire) busy_d  = 1'b1;
      else if (busy_q && last_term) begin
        valid_d = 1'b1;
        busy_d  = 1'b0;
      end
    end
    // deassert valid on out_fire
    if (out_fire) begin
      valid_d = 1'b0;
    end
  end

  /* ------------------------ Data Unpacking and Sequencing ------------------------ */
  logic signed [TotalTerms-1:0][InBits-1:0]     flat_windows;
  logic signed [TotalTerms-1:0][WeightBits-1:0] flat_weights;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_flat_ch
      for (genvar k = 0; k < KernelArea; k++) begin : gen_flat_k
        assign flat_windows[ch*KernelArea + k] = windows_q[ch][k];
        assign flat_weights[ch*KernelArea + k] = weights_q[ch][k];
      end
    end
  endgenerate

  // term_idx is 0 during fire cycle, then follows term_count_q during busy cycles
  wire [TermBits-1:0] term_idx = in_fire ? '0 : term_count_q;

  // Bypass logic
  // Use raw input during fire cycle to save a cycle of latency, otherwise use captured registers
  wire signed [InBits-1:0]     data_w   = in_fire ? windows_i[0][0] : flat_windows[term_idx];
  wire signed [WeightBits-1:0] weight_w = in_fire ? weights_i[0][0] : flat_weights[term_idx];

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
    ,.en_i        (in_fire | busy_q)
    ,.load_bias_i (in_fire)
    ,.data_i      (data_w)
    ,.weight_i    (weight_w)
    ,.bias_i      (bias_i)
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
