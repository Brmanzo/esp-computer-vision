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

  // Capture registers for isolation
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

    if (in_fire) begin
      // If neuron processing a single term, output is immediately valid
      if (single_term) valid_d = 1'b1;
      else             busy_d  = 1'b1;
    end else if (busy_q) begin
      // When on last term, no longer busy 
      if (last_term) begin
        valid_d = 1'b1;
        busy_d  = 1'b0;
      end
    end

    if (out_fire) begin
      valid_d = 1'b0;
    end
  end

  /* ----------------------------------- DSP Mapping ----------------------------------- */
  logic signed [TotalTerms-1:0][InBits-1:0]     flat_windows;
  logic signed [TotalTerms-1:0][WeightBits-1:0] flat_weights;

  genvar g_ch, g_k;
  generate
    for (g_ch = 0; g_ch < int'(InChannels); g_ch++) begin : gen_flat_ch
      for (g_k = 0; g_k < int'(KernelArea); g_k++) begin : gen_flat_k
        assign flat_windows[g_ch*KernelArea + g_k] = windows_q[g_ch][g_k];
        assign flat_weights[g_ch*KernelArea + g_k] = weights_q[g_ch][g_k];
      end
    end
  endgenerate

  // mux_idx is 0 during fire cycle, then follows term_count_q during busy cycles
  wire [TermBits-1:0] mux_idx = in_fire ? '0 : term_count_q;

  // Bypass logic
  // Use raw input during fire cycle to save a cycle of latency, otherwise use captured registers
  wire signed [InBits-1:0]     raw_data_w   = in_fire ? windows_i[0][0] : flat_windows[mux_idx];
  wire signed [WeightBits-1:0] raw_weight_w = in_fire ? weights_i[0][0] : flat_weights[mux_idx];
 
  /* --------------------------------- Input Encoding --------------------------------- */
  // Encode inputs for DSP (Bipolar for 1-bit, Ternary for 2-bit)
  wire signed [`SB_MAC16_IN-1:0] data_w;
  generate
    // Binary encoding {0,1} -> {-1,1}
    if (InBits == 1) begin : gen_binary_in
      assign data_w = (raw_data_w[0]) ? `SB_MAC16_IN'sd1 : -`SB_MAC16_IN'sd1;
    // Ternary encoding {-1,0,1}
    end else if (InBits == 2) begin : gen_ternary_in
      assign data_w = (raw_data_w == 2'sb01) ?  `SB_MAC16_IN'sd1 : 
                      (raw_data_w == 2'sb11) ? -`SB_MAC16_IN'sd1 : 
                                                `SB_MAC16_IN'sd0;
    // Otherwise sign extend to 16b input
    end else begin : gen_full_in
      assign data_w = `SB_MAC16_IN'($signed(raw_data_w));
    end
  endgenerate

  // Encode weights
  wire signed [`SB_MAC16_IN-1:0] weight_w;
  generate
    // Binary encoding {0,1} -> {-1,1}
    if (WeightBits == 1) begin : gen_binary_weight
      assign weight_w = (raw_weight_w[0]) ? `SB_MAC16_IN'sd1 : -`SB_MAC16_IN'sd1;
    // Ternary encoding {-1,0,1}
    end else if (WeightBits == 2) begin : gen_ternary_weight
      assign weight_w = (raw_weight_w == 2'sb01) ? `SB_MAC16_IN'sd1 : 
                        (raw_weight_w == 2'sb11) ? -`SB_MAC16_IN'sd1 : 
                                                   `SB_MAC16_IN'sd0;
    end else begin : gen_full_weight
      assign weight_w = `SB_MAC16_IN'($signed(raw_weight_w));
    end
  endgenerate

  /* ------------------------------- Neuron DSP Unit ------------------------------- */
  wire signed [AccBits-1:0] acc_o;

  neuron_dsp #(
    .InBits    (`SB_MAC16_IN),
    .WeightBits(`SB_MAC16_IN),
    .BiasBits  (AccBits),
    .OutBits   (AccBits)
  ) dsp_inst (
    .clk_i      (clk_i),
    .rst_i      (rst_i),
    .en_i       (in_fire | busy_q),
    .load_bias_i(in_fire),
    .data_i     (data_w),
    .weight_i   (weight_w),
    .bias_i     (bias_i),
    .acc_o      (acc_o)
  );

  /* -------------------------------- Output Encoding -------------------------------- */
  generate
    // Binary Output Encoding {-1,1} -> {0,1}
    if (OutBits == 1) begin : gen_binary_out
      assign data_o = (acc_o > 0) ? OutBits'(1) : OutBits'(0);
    // Ternary Output Encoding {-1,0,1}
    end else if (OutBits == 2) begin : gen_ternary_out
      assign data_o = (acc_o > 0) ? OutBits'(1) :
                      (acc_o < 0) ? OutBits'(-1) :
                                    OutBits'(0);
    // Full or Extended Output
    end else if (OutBits >= AccBits) begin : gen_full_out
      assign data_o = OutBits'($signed(acc_o));
    // Truncated MSB Output
    end else begin : gen_truncated_out
      assign data_o = acc_o[AccBits-1 -: OutBits];
    end
  endgenerate
endmodule
