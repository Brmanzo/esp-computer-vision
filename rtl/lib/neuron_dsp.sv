// rtl/lib/neuron_dsp.sv
// Bradley Manzo 2026
// Optimized for iCE40 UltraPlus SB_MAC16 inference

`timescale 1ns / 1ps
module neuron_dsp #(
    parameter int unsigned InBits     = 16
   ,parameter int unsigned WeightBits = 2
   ,parameter int unsigned AccBits    = 32
) (
    input  logic [0:0] clk_i
   ,input  logic [0:0] rst_i
   ,input  logic [0:0] en_i
   ,input  logic [0:0] load_bias_i
   ,input  logic signed [InBits-1:0]     data_i
   ,input  logic signed [WeightBits-1:0] weight_i
   ,input  logic signed [AccBits-1:0]    bias_i
   ,output logic signed [AccBits-1:0]    acc_o
);

    // Force DSP inference by widening the operands to 16 bits.
    // iCE40 DSPs (SB_MAC16) are 16x16 multipliers. If the inputs are too narrow (e.g. 1-bit),
    // Yosys heuristics will often implement the logic in LUTs instead of "wasting" a DSP.
    // By explicitly using 16-bit signed types here, we trigger the DSP mapping.
    wire signed [15:0] data_w   = 16'($signed(data_i));
    wire signed [15:0] weight_w = 16'($signed(weight_i));

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            acc_o <= '0;
        end else if (en_i) begin
            if (load_bias_i) begin
                // On the first cycle of a frame, load the bias and perform the first multiplication
                acc_o <= $signed(bias_i) + (data_w * weight_w);
            end else begin
                // On subsequent cycles, accumulate the product
                acc_o <= $signed(acc_o) + (data_w * weight_w);
            end
        end
    end

endmodule
