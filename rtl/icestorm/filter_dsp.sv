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

  ,input logic signed [InChannels*KernelArea*InBits-1:0]     windows_i
  ,output [TermBits-1:0]                                    term_idx_o
  ,input logic signed [WeightBits-1:0]                      weight_i
  ,input logic signed [BiasBits-1:0]                        bias_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutBits-1:0] data_o
);

  /* ------------------------ Internal Signals ------------------------ */
  wire [0:0] in_fire  = valid_i && ready_o;
  wire [0:0] out_fire = valid_o && ready_i;

  /* -------------------------- Control Logic -------------------------- */
  logic [TermBits-1:0] term_count_q;
  logic [0:0]          busy_d,  busy_q;
  logic [0:0]          valid_d, valid_q;
  wire [0:0]           done;
  
  assign valid_o = valid_q;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      busy_q    <= 1'b0;
      valid_q   <= 1'b0;
    end else begin
      busy_q       <= busy_d;
      valid_q      <= valid_d;
    end
  end

    wire [0:0] last_term = (term_count_q == TermBits'(TotalTerms - 1));
    assign done      = busy_q && last_term;
    
    // Block new input if we are busy OR if we are holding a result that hasn't been accepted
    assign ready_o = ~busy_q && (~valid_q || ready_i);

    counter_roll #(
       .CountBits  (TermBits)
      ,.ResetVal   (0)
      ,.MaxVal     (TotalTerms - 1)
      ,.EnableDown (1'b0)
    ) term_counter_inst (
       .clk_i      (clk_i)
      ,.rst_i      (rst_i | in_fire)
      ,.up_i       (busy_q)
      ,.down_i     (1'b0)
      ,.count_o    (term_count_q)
    );

    always_comb begin
      busy_d  = busy_q;
      valid_d = valid_q;
      if (in_fire) begin
        busy_d  = 1'b1;
      end else if (done) begin
        valid_d = 1'b1;
        busy_d  = 1'b0;
      end
      if (out_fire) valid_d = 1'b0;
    end

  /* ------------------------ Data Sequencing ------------------------ */
  // term_idx follows term_count_q during busy cycles
  wire [TermBits-1:0] term_idx = term_count_q;
  assign term_idx_o = term_idx;

  // Access weights and windows directly without registration
  wire signed [InBits-1:0]     data_w   = windows_i[term_idx*InBits +: InBits];
  wire signed [WeightBits-1:0] weight_w = weight_i;

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
      ,.en_i        (busy_q)
      ,.load_bias_i (busy_q && (term_count_q == '0))
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
