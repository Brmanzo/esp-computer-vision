// Bradley Manzo, 2026

`timescale 1ns / 1ps
module tb_single_block #(
   parameter int unsigned C_LineWidthPx = 16
  ,parameter int unsigned C_LineCountPx = 12
  ,parameter int unsigned C_InBits      = 1
  ,parameter int unsigned C_OutBits     = 1
  ,parameter int unsigned C_KernelWidth = 3
  ,parameter int unsigned P_KernelWidth = 2
  ,parameter int unsigned C_WeightBits  = 2
  ,parameter int unsigned C_InChannels  = 1
  ,parameter int unsigned C_OutChannels = 1
  ,parameter int unsigned C_Stride      = 1
  ,parameter int unsigned P_Mode        = 0 // 0 for max pooling, 1 for average pooling
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [C_InChannels-1:0][C_InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [C_OutChannels-1:0][C_OutBits-1:0] data_o
);

  // Calculate reduced input dimensions for pooling layer based on convolutional layer parameters.
  function automatic int unsigned p_in_dim;
    input int unsigned c_dim, c_kernel, c_stride;
    begin
      p_in_dim = ((c_dim - c_kernel) / c_stride) + 1;
    end
  endfunction

  localparam int unsigned P_LineWidthPx = p_in_dim(C_LineWidthPx, C_KernelWidth, C_Stride);
  localparam int unsigned P_LineCountPx = p_in_dim(C_LineCountPx, C_KernelWidth, C_Stride);

  `include "injected_weights.vh"

  wire [0:0] pool_0_ready;
  wire [0:0] conv_0_valid;
  wire [C_OutBits*C_OutChannels-1:0] conv_0_data;

  conv_layer #(
     .LineWidthPx (C_LineWidthPx)
    ,.LineCountPx (C_LineCountPx)
    ,.InBits      (C_InBits)
    ,.OutBits     (C_OutBits)
    ,.KernelWidth (C_KernelWidth)
    ,.WeightBits  (C_WeightBits)
    ,.InChannels  (C_InChannels)
    ,.OutChannels (C_OutChannels)
    ,.Stride      (C_Stride)
    ,.Weights     (INJECTED_WEIGHTS)
  ) conv_layer_inst_0 (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o  (ready_o)
    ,.valid_i  (valid_i)
    ,.data_i   (data_i)

    ,.ready_i  (pool_0_ready)
    ,.valid_o  (conv_0_valid)
    ,.data_o   (conv_0_data)
  );

  pool_layer #(
     .LineWidthPx (P_LineWidthPx) // Reduction in size due to
    ,.LineCountPx (P_LineCountPx) // conv_layer kernel width and stride.
    ,.InBits      (C_OutBits)     // Same bitwidth as conv_layer output.
    ,.KernelWidth (P_KernelWidth) // Power of 2 for proper average pooling.
    ,.InChannels  (C_OutChannels)  // Same as conv_layer out channels.
    ,.PoolMode    (P_Mode)              // Max pooling
  ) pool_layer_inst_0 (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)

    ,.ready_o  (pool_0_ready)
    ,.valid_i  (conv_0_valid)
    ,.data_i   (conv_0_data)

    ,.ready_i  (ready_i)
    ,.valid_o  (valid_o)
    ,.data_o   (data_o)
  );

endmodule
