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

  /* ---------------------------- Input Logic ---------------------------- */
  // Binary input encoding {0,1} -> {-1,1}
  wire signed [`SB_MAC16_IN-1:0] data_w;
  generate
    if (InBits == 1) begin: gen_binary_in
      assign data_w = (data_i[0]) ? `SB_MAC16_IN'sd1 : -`SB_MAC16_IN'sd1;
    end else begin : gen_full_in
      assign data_w = `SB_MAC16_IN'($signed(data_i));
    end
  endgenerate 

  wire signed [`SB_MAC16_IN-1:0] weight_w = `SB_MAC16_IN'($signed(weight_i));
  
  // Internal 32-bit accumulator for SB_MAC16
  logic signed [`SB_MAC16_OUT-1:0] acc_r;

  /* ----------------------------- Output Logic ----------------------------- */
  generate
    // Binary Output Encoding {-1,1} -> {0,1}
    if (OutBits == 1) begin : gen_binary_out
      assign acc_o = (acc_r > $signed('0)) ? 1'b1 : 1'b0;
    // Ternary Output Encoding {-1,0,1}
    end else if (OutBits == 2) begin : gen_ternary_out
      assign acc_o = (acc_r > $signed('0)) ? OutBits'( 1) :
                     (acc_r < $signed('0)) ? OutBits'(-1) :
                                             OutBits'( 0);
    // Linear Output
    end else if (OutBits == `SB_MAC16_OUT) begin : gen_full_out
      assign acc_o = acc_r;
    // Return the LSBs to match standard neuron.sv behavior for small OutBits
    end else begin : gen_truncated_out
      // Output: Take MSB threshold for 1-bit, else slice LSBs
      assign acc_o = (OutBits == 1) ? OutBits'(`SB_MAC16_OUT'($signed(acc_r)) > 0) : acc_r[OutBits-1:0];
    end
  endgenerate

  /* ----------------------------- Sequential Accumulator Logic ----------------------------- */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      acc_r <= '0;
    end else if (en_i) begin
      if (load_bias_i) begin
        // On the first cycle of a frame, load the bias and perform the first multiplication
        acc_r <= `SB_MAC16_OUT'($signed(bias_i)) + `SB_MAC16_OUT'($signed(data_w * weight_w));
      end else begin
        // On subsequent cycles, accumulate the product
        acc_r <= acc_r + `SB_MAC16_OUT'($signed(data_w * weight_w));
      end
    end
  end

endmodule
