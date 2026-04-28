// Bradley Manzo, 2026

`timescale 1ns / 1ps
`include "injected_weights.vh" // Include at the very top

module tb_double_block #(
   // Parameters default to the macros injected by the Python testbench
   parameter int unsigned C0_LineWidthPx = `C0_LineWidthPx
  ,parameter int unsigned C0_LineCountPx = `C0_LineCountPx
  ,parameter int unsigned C0_InBits      = `C0_InBits
  ,parameter int unsigned C0_OutBits     = `C0_OutBits
  ,parameter int unsigned C0_KernelWidth = `C0_KernelWidth
  ,parameter int unsigned C0_WeightBits  = `C0_WeightBits
  ,parameter int unsigned C0_InChannels  = `C0_InChannels
  ,parameter int unsigned C0_OutChannels = `C0_OutChannels
  ,parameter int unsigned C0_Stride      = `C0_Stride
  ,parameter int unsigned C0_Padding     = `C0_Padding

  ,parameter int unsigned P0_KernelWidth = `P0_KernelWidth
  ,parameter int unsigned P0_Mode        = `P0_Mode 

  ,parameter int unsigned C1_OutBits     = `C1_OutBits
  ,parameter int unsigned C1_KernelWidth = `C1_KernelWidth
  ,parameter int unsigned C1_WeightBits  = `C1_WeightBits
  ,parameter int unsigned C1_OutChannels = `C1_OutChannels
  ,parameter int unsigned C1_Stride      = `C1_Stride
  ,parameter int unsigned C1_Padding     = `C1_Padding
  
  ,parameter int unsigned P1_KernelWidth = `P1_KernelWidth
  ,parameter int unsigned P1_Mode        = `P1_Mode 
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [C0_InChannels-1:0][C0_InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [C1_OutChannels-1:0][C1_OutBits-1:0] data_o
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

  localparam int unsigned P1_LineWidthPx  = conv_out_dim(C1_LineWidthPx, C1_KernelWidth, C1_Stride, C1_Padding);
  localparam int unsigned P1_LineCountPx  = conv_out_dim(C1_LineCountPx, C1_KernelWidth, C1_Stride, C1_Padding); 

  // Wires (Using 2D arrays to match your ports cleanly)
  wire [0:0] c0_valid;
  logic signed [C0_OutChannels-1:0][C0_OutBits-1:0] c0_data;

  wire [0:0] p0_ready;
  wire [0:0] p0_valid;
  logic signed [C0_OutChannels-1:0][C0_OutBits-1:0] p0_data;

  wire [0:0] c1_ready;
  wire [0:0] c1_valid;
  logic signed [C1_OutChannels-1:0][C1_OutBits-1:0] c1_data;

  wire [0:0] p1_ready;

  conv_layer #(
     .LineWidthPx (C0_LineWidthPx)
    ,.LineCountPx (C0_LineCountPx)
    ,.InBits      (C0_InBits)
    ,.OutBits     (C0_OutBits)
    ,.KernelWidth (C0_KernelWidth)
    ,.WeightBits  (C0_WeightBits)
    ,.InChannels  (C0_InChannels)
    ,.OutChannels (C0_OutChannels)
    ,.Stride      (C0_Stride)
    ,.Padding     (C0_Padding)
    ,.Weights     (INJECTED_WEIGHTS_0) // Fixed Name
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
    ,.InChannels  (C0_OutChannels)
    ,.OutChannels (C1_OutChannels)
    ,.Stride      (C1_Stride)
    ,.Padding     (C1_Padding)
    ,.Weights     (INJECTED_WEIGHTS_1) // Fixed Name
  ) conv_layer_inst_1 (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o  (c1_ready)
    ,.valid_i  (p0_valid)
    ,.data_i   (p0_data)

    ,.ready_i  (p1_ready)
    ,.valid_o  (c1_valid)
    ,.data_o   (c1_data)
  );

  pool_layer #(
     .LineWidthPx (P1_LineWidthPx) 
    ,.LineCountPx (P1_LineCountPx) 
    ,.InBits      (C1_OutBits)     
    ,.KernelWidth (P1_KernelWidth) 
    ,.InChannels  (C1_OutChannels)  
    ,.PoolMode    (P1_Mode)              
  ) pool_layer_inst_1 (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)

    ,.ready_o  (p1_ready)
    ,.valid_i  (c1_valid)
    ,.data_i   (c1_data)

    ,.ready_i  (ready_i)
    ,.valid_o  (valid_o)
    ,.data_o   (data_o)
  );

endmodule