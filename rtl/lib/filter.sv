// filter.sv
// Bradley Manzo, 2026

`timescale 1ns/1ps
module filter #(
   parameter   int unsigned WidthIn      = 1
  ,parameter   int unsigned WidthOut     = 32
  ,parameter   int unsigned KernelWidth  = 3
  ,parameter   int unsigned WeightWidth  = 2
  ,parameter   int unsigned InChannels   = 2
  ,localparam  int unsigned KernelArea   = KernelWidth * KernelWidth
)  (
   input [InChannels-1:0][KernelArea-1:0][WidthIn-1:0] windows_i
  ,input  signed [InChannels-1:0][KernelArea-1:0][WeightWidth-1:0] weights_i
  
  ,output logic signed [WidthOut-1:0] data_o
);

  /* ------------------------------------ Output Channels ------------------------------------ */
  logic signed [InChannels-1:0][WidthOut-1:0] kernel_data_o;
  logic signed [WidthOut-1:0] sum_d;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_InChannels
      mac #(
         .KernelWidth(KernelWidth)
        ,.WidthIn    (WidthIn)
        ,.WidthOut   (WidthOut)
        ,.WeightWidth(WeightWidth)
      ) mac_inst (
         .window   (windows_i[ch])
        ,.weights_i(weights_i[ch])
        ,.data_o   (kernel_data_o[ch])
      );
    end
  endgenerate

  /* ----------------------------- Filter Output Summation Logic ----------------------------- */
  always_comb begin
    sum_d = '0;
    for (int ch = 0; ch < InChannels; ch++) begin
      sum_d += kernel_data_o[ch];
    end
  end

  assign data_o = sum_d;

endmodule
