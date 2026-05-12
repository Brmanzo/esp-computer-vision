// rtl/lib/neuron_dsp.sv
// Bradley Manzo 2026

// Targetting IceStorm's 8 SB_MAC16 DSPs
// https://0x04.net/~mwk/sbdocs/ice40/FPGA-TN-02007-1-2-DSP-Function-Usage-Guide-for-iCE40-Devices.pdf
`define SB_MAC16_IN  16
`define SB_MAC16_OUT 32

`timescale 1ns / 1ps
module neuron_dsp #(
   parameter int unsigned InBits     = `SB_MAC16_IN
  ,parameter int unsigned OutBits    = `SB_MAC16_OUT
  ,parameter int unsigned WeightBits = 2
  ,parameter int unsigned BiasBits   = `SB_MAC16_OUT
) (
   input  logic [0:0] clk_i
  ,input  logic [0:0] rst_i

  ,input  logic [0:0] en_i
  ,input  logic [0:0] load_bias_i
  
  ,input  logic signed [InBits-1:0]     data_i
  ,input  logic signed [WeightBits-1:0] weight_i
  ,input  logic signed [BiasBits-1:0]   bias_i

  ,output logic signed [OutBits-1:0]    acc_o
);

  /* ------------------------ Binary/Ternary input encoding ------------------------ */
  logic signed [`SB_MAC16_IN-1:0] data_w;
  logic signed [`SB_MAC16_IN-1:0] weight_w;

  /* verilator lint_off WIDTHEXPAND */
  always_comb begin
    if (InBits == 1) begin
      data_w = data_i[0] ? `SB_MAC16_IN'(1) : `SB_MAC16_IN'(-1);
    end else if (InBits == 2) begin
      data_w = (data_i == 2'sb01) ? `SB_MAC16_IN'( 1) :
               (data_i == 2'sb11) ? `SB_MAC16_IN'(-1) :
                                    `SB_MAC16_IN'( 0);
    end else begin
      data_w = `SB_MAC16_IN'($signed(data_i));
    end
  end
  /* ------------------------ Binary/Ternary weight encoding ------------------------ */
  always_comb begin
    if (WeightBits == 1) begin
      weight_w = weight_i[0] ? `SB_MAC16_IN'(1) : `SB_MAC16_IN'(-1);
    end else if (WeightBits == 2) begin
      weight_w = (weight_i == 2'sb01) ? `SB_MAC16_IN'( 1) :
                 (weight_i == 2'sb11) ? `SB_MAC16_IN'(-1) :
                                        `SB_MAC16_IN'( 0);
    end else begin
      weight_w = `SB_MAC16_IN'($signed(weight_i));
    end
  end
  /* verilator lint_on WIDTHEXPAND */
  
  // Internal 32-bit accumulator for SB_MAC16
  logic signed [`SB_MAC16_OUT-1:0] acc_r;
  /* verilator lint_off WIDTHTRUNC */
  assign acc_o = acc_r;
  /* verilator lint_on WIDTHTRUNC */

  /* -------------------------- Multiply and Accumulate via SB_MAC16 -------------------------- */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      acc_r <= '0;
    end else if (en_i) begin
      if (load_bias_i) begin
        // On the first cycle of a frame, load the bias and perform the first multiplication
        acc_r <= `SB_MAC16_OUT'($signed(bias_i)) + (`SB_MAC16_OUT'($signed(data_w)) * `SB_MAC16_OUT'($signed(weight_w)));
      end else begin
        // On subsequent cycles, accumulate the product
        acc_r <= acc_r + (`SB_MAC16_OUT'($signed(data_w)) * `SB_MAC16_OUT'($signed(weight_w)));
      end
    end
  end

endmodule
