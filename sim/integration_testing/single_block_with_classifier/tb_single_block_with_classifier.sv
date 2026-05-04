// Bradley Manzo, 2026

`timescale 1ns / 1ps
`include "injected_weights_0.vh"
`include "injected_biases_0.vh"
`include "injected_weights_1.vh"
`include "injected_biases_1.vh"
`include "injected_weights_2.vh"
`include "injected_biases_2.vh"

module tb_single_block_with_classifier #(
   // Parameters default to the macros injected by the Python testbench
   parameter int unsigned C0_LineWidthPx = 32
  ,parameter int unsigned C0_LineCountPx = 24
  ,parameter int unsigned C0_InBits      = 1
  ,parameter int unsigned C0_OutBits     = 1
  ,parameter int unsigned C0_KernelWidth = 3
  ,parameter int unsigned C0_WeightBits  = 2
  ,parameter int unsigned C0_BiasBits    = 8
  ,parameter int unsigned C0_InChannels  = 1
  ,parameter int unsigned C0_OutChannels = 2
  ,parameter int unsigned C0_Stride      = 1
  ,parameter int unsigned C0_Padding     = 1

  ,parameter int unsigned P0_KernelWidth = 2
  ,parameter int unsigned P0_Mode        = 0

  ,parameter int unsigned C1_OutBits     = 1
  ,parameter int unsigned C1_KernelWidth = 3
  ,parameter int unsigned C1_WeightBits  = 2
  ,parameter int unsigned C1_BiasBits    = 8
  ,parameter int unsigned C1_OutChannels = 4
  ,parameter int unsigned C1_Stride      = 1
  ,parameter int unsigned C1_Padding     = 1
  
  ,parameter int unsigned ClassCount     = 10
  ,parameter int unsigned BusBits        = 8
  ,parameter int unsigned ClassWeightBits = 2
  ,parameter int unsigned ClassBiasBits   = 8
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [C0_InChannels-1:0][C0_InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [BusBits-1:0] data_o
);

  // 1. Calculate output dimensions of Convolution Layers
  function automatic int unsigned conv_out_dim;
    input int unsigned c_dim, c_kernel, c_stride, c_padding;
    begin
      // Padding is added BEFORE stride division!
      conv_out_dim = ((c_dim + 2 * c_padding - c_kernel) / c_stride) + 1; 
    end
  endfunction

  // 2. Calculate output dimensions of Pooling Layers
  function automatic int unsigned pool_out_dim;
    input int unsigned p_dim, p_kernel;
    begin
      // Standard pooling stride matches the kernel size
      pool_out_dim = ((p_dim - p_kernel) / p_kernel) + 1; 
    end
  endfunction

  // Dimension Routing
  localparam int unsigned P0_LineWidthPx  = conv_out_dim(C0_LineWidthPx, C0_KernelWidth, C0_Stride, C0_Padding);
  localparam int unsigned P0_LineCountPx  = conv_out_dim(C0_LineCountPx, C0_KernelWidth, C0_Stride, C0_Padding);

  // Input to Conv1 is the OUTPUT of Pool0
  localparam int unsigned C1_LineWidthPx  = pool_out_dim(P0_LineWidthPx, P0_KernelWidth);
  localparam int unsigned C1_LineCountPx  = pool_out_dim(P0_LineCountPx, P0_KernelWidth);

  localparam int unsigned ClassifierTermCount = conv_out_dim(C1_LineWidthPx, C1_KernelWidth, C1_Stride, C1_Padding) * conv_out_dim(C1_LineCountPx, C1_KernelWidth, C1_Stride, C1_Padding);

  // Wires (Using 2D arrays to match your ports cleanly)
  wire [0:0] c0_valid;
  logic signed [C0_OutChannels-1:0][C0_OutBits-1:0] c0_data;

  wire [0:0] p0_ready;
  wire [0:0] p0_valid;
  logic signed [C0_OutChannels-1:0][C0_OutBits-1:0] p0_data;

  wire [0:0] c1_ready;
  wire [0:0] c1_valid;
  logic signed [C1_OutChannels-1:0][C1_OutBits-1:0] c1_data;

  wire [0:0] class_ready;

  conv_layer #(
     .LineWidthPx (C0_LineWidthPx)
    ,.LineCountPx (C0_LineCountPx)
    ,.InBits      (C0_InBits)
    ,.OutBits     (C0_OutBits)
    ,.KernelWidth (C0_KernelWidth)
    ,.WeightBits  (C0_WeightBits)
    ,.BiasBits    (C0_BiasBits)
    ,.InChannels  (C0_InChannels)
    ,.OutChannels (C0_OutChannels)
    ,.Stride      (C0_Stride)
    ,.Padding     (C0_Padding)
    ,.Weights     (INJECTED_WEIGHTS_0)
    ,.Biases      (INJECTED_BIASES_0)
  ) conv_layer_inst_0 (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o  (ready_o)
    ,.valid_i  (valid_i)
    ,.data_i   (data_i)

    ,.ready_i  (p0_ready)
    ,.valid_o  (c0_valid)
    ,.data_o   (c0_data)
  );

  pool_layer #(
     .LineWidthPx (P0_LineWidthPx) 
    ,.LineCountPx (P0_LineCountPx) 
    ,.InBits      (C0_OutBits)     
    ,.KernelWidth (P0_KernelWidth) 
    ,.InChannels  (C0_OutChannels)  
    ,.PoolMode    (P0_Mode)              
  ) pool_layer_inst_0 (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)

    ,.ready_o  (p0_ready)
    ,.valid_i  (c0_valid)
    ,.data_i   (c0_data)

    ,.ready_i  (c1_ready)
    ,.valid_o  (p0_valid)
    ,.data_o   (p0_data)
  );

  conv_layer #(
     .LineWidthPx (C1_LineWidthPx) // Fixed: Uses Post-Pool dimensions
    ,.LineCountPx (C1_LineCountPx) // Fixed: Uses Post-Pool dimensions
    ,.InBits      (C0_OutBits)
    ,.OutBits     (C1_OutBits)
    ,.KernelWidth (C1_KernelWidth)
    ,.WeightBits  (C1_WeightBits)
    ,.BiasBits    (C1_BiasBits)
    ,.InChannels  (C0_OutChannels)
    ,.OutChannels (C1_OutChannels)
    ,.Stride      (C1_Stride)
    ,.Padding     (C1_Padding)
    ,.Weights     (INJECTED_WEIGHTS_1)
    ,.Biases      (INJECTED_BIASES_1)
  ) conv_layer_inst_1 (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o  (c1_ready)
    ,.valid_i  (p0_valid)
    ,.data_i   (p0_data)

    ,.ready_i  (class_ready)
    ,.valid_o  (c1_valid)
    ,.data_o   (c1_data)
  );

  classifier_layer #(
     .TermBits   (C1_OutBits)
    ,.TermCount  (ClassifierTermCount)
    ,.BusBits    (BusBits)
    ,.InChannels (C1_OutChannels)
    ,.ClassCount (ClassCount)
    ,.WeightBits (ClassWeightBits)
    ,.BiasBits   (ClassBiasBits)
    ,.Weights    (INJECTED_WEIGHTS_2)
    ,.Biases     (INJECTED_BIASES_2)
  ) classifier_layer_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o  (class_ready)
    ,.valid_i  (c1_valid)
    ,.data_i   (c1_data)

    ,.valid_o  (valid_o)
    ,.ready_i  (ready_i)
    ,.class_o  (data_o)
  );

endmodule