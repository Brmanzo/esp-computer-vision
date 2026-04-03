// filter.sv
// Bradley Manzo, 2026

`timescale 1ns/1ps
module filter #(
   parameter   int unsigned InBits      = 1
  ,parameter   int unsigned OutBits     = 1
  ,parameter   int unsigned KernelWidth = 3
  ,parameter   int unsigned WeightBits  = 2
  ,parameter   int unsigned MacBits     = 32
  ,parameter   int unsigned AccBits     = 32
  ,parameter   int unsigned InChannels  = 2
  ,localparam  int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input [InChannels-1:0][KernelArea-1:0][InBits-1:0] windows_i
  ,input  signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_i
  
  ,output logic signed [OutBits-1:0] data_o
);

  /* ------------------------------------ Output Channels ------------------------------------ */
  logic signed [InChannels-1:0][MacBits-1:0] kernel_data_o;
  logic signed [AccBits-1:0] sum_d;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_InChannels
      mac #(
         .KernelWidth(KernelWidth)
        ,.InBits    (InBits)
        ,.OutBits   (OutBits)
        ,.AccBits   (MacBits)
        ,.WeightBits(WeightBits)
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
    // If binary activation, encode a positive sum as 1 and a negative sum as 0
    if (OutBits == 1) begin
      data_o = (sum_d > 0) ? OutBits'(1) : OutBits'(0);
    end else begin
      data_o = OutBits'(sum_d);
    end
  end

endmodule
