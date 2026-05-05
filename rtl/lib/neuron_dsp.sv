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

    // This behavioral pattern is designed to trigger SB_MAC16 inference in Yosys
    // It performs a Multiply-Accumulate (MAC) with an optional initial bias load.
    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            acc_o <= '0;
        end else if (en_i) begin
            if (load_bias_i) begin
                // On the first cycle of a frame, load the bias and perform the first multiplication
                acc_o <= AccBits'($signed(bias_i) + ($signed(data_i) * $signed(weight_i)));
            end else begin
                // On subsequent cycles, accumulate the product
                acc_o <= AccBits'($signed(acc_o) + ($signed(data_i) * $signed(weight_i)));
            end
        end
    end

endmodule
