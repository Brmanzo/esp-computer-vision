// filter_dsp.sv
// Bradley Manzo, 2026

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
   ,parameter   logic signed [BiasBits-1:0] Bias = '0

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

   ,output [0:0] valid_o
   ,input  [0:0] ready_i
   ,output logic signed [OutBits-1:0] data_o
);

    /* ------------------------ Sequential Logic ------------------------ */
    logic [TermBits-1:0] term_counter;
    logic [0:0] busy_q, valid_r;
    
    // Capture registers for the parallel window and weights
    logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_q;
    // Weights are usually parameters, but we capture them to simplify the mux
    logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_q;

    wire first_term = (term_counter == '0);
    wire last_term  = (term_counter == TermBits'(TotalTerms - 1));

    assign ready_o = ~busy_q & ~valid_r;
    assign valid_o = valid_r;

    wire in_fire  = valid_i && ready_o;
    wire out_fire = valid_o && ready_i;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            term_counter <= '0;
            busy_q       <= 1'b0;
            valid_r      <= 1'b0;
        end else begin
            if (in_fire) begin
                if (TotalTerms > 1) begin
                    busy_q       <= 1'b1;
                    term_counter <= 1; // Start from second term on next cycle
                end else begin
                    valid_r      <= 1'b1;
                end
            end else if (busy_q) begin
                if (term_counter == TermBits'(TotalTerms - 1)) begin
                    busy_q  <= 1'b0;
                    valid_r <= 1'b1;
                end else begin
                    term_counter <= term_counter + 1;
                end
            end

            if (out_fire) begin
                valid_r <= 1'b0;
            end
        end
    end

    /* ------------------------- DSP Mapping ------------------------- */
    // Flatten the input data for the multiplexer. This is safe because
    // the conv_layer stalls and holds the window stable while busy_q is high.
    wire signed [TotalTerms-1:0][InBits-1:0]     flat_windows = windows_i;
    wire signed [TotalTerms-1:0][WeightBits-1:0] flat_weights = weights_i;

    /* ------------------------ Serialization Logic ------------------------ */
    wire [TermBits-1:0] mux_idx = in_fire ? '0 : term_counter;
    wire signed [InBits-1:0]     raw_data_w   = in_fire ? windows_i[0][0] : flat_windows[mux_idx];
    wire signed [WeightBits-1:0] raw_weight_w = in_fire ? weights_i[0][0] : flat_weights[mux_idx];

    // Encode terms for DSP (Bipolar for 1-bit, Ternary for 2-bit)
    wire signed [`SB_MAC16_IN-1:0] data_w = 
        (InBits == 1) ? (raw_data_w[0] ? `SB_MAC16_IN'sd1 : -`SB_MAC16_IN'sd1) : 
        (InBits == 2) ? ((raw_data_w == InBits'(1)) ? `SB_MAC16_IN'sd1 : (raw_data_w ==  InBits'(-1)) ? -`SB_MAC16_IN'sd1 : `SB_MAC16_IN'sd0) :
                        `SB_MAC16_IN'($signed(raw_data_w));

    wire signed [`SB_MAC16_IN-1:0] weight_w = 
        (WeightBits == 1) ? (raw_weight_w[0] ? `SB_MAC16_IN'sd1 : -`SB_MAC16_IN'sd1) : 
        (WeightBits == 2) ? ((raw_weight_w == 2'sb01) ? `SB_MAC16_IN'sd1 : (raw_weight_w == 2'sb11) ? -`SB_MAC16_IN'sd1 : `SB_MAC16_IN'sd0) :
                            `SB_MAC16_IN'($signed(raw_weight_w));

    wire signed [AccBits-1:0] acc_o;

    // Single DSP core for multiplying and accumulating the filter's terms
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
        .bias_i     (AccBits'($signed(Bias))),
        .acc_o      (acc_o)
    );

    /* ------------------------- Output Activation ------------------------- */
    generate
        if (OutBits == 1) begin : gen_binary_out
            assign data_o = (acc_o > 0) ? OutBits'(1) : OutBits'(0);
        end else if (OutBits == 2) begin : gen_ternary_out
            assign data_o = (acc_o > 0) ? OutBits'(1) :
                            (acc_o < 0) ? OutBits'(-1) :
                                          OutBits'(0);
        end else if (OutBits >= AccBits) begin : gen_full_or_extended_out
            assign data_o = OutBits'($signed(acc_o));
        end else begin : gen_truncated_out
            // MSB Selection / Bit-slicing
            assign data_o = acc_o[AccBits-1 -: OutBits];
        end
    endgenerate

endmodule
