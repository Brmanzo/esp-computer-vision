// filter.sv
// Bradley Manzo, 2026

`timescale 1ns/1ps
module filter #(
   parameter   int unsigned InBits      = 1
  ,parameter   int unsigned OutBits     = 1
  ,parameter   int unsigned KernelWidth = 3
  ,parameter   int unsigned WeightBits  = 2
  ,parameter   int unsigned BiasBits    = 8
  ,parameter   int unsigned AccBits     = 32
  ,parameter   int unsigned ShiftBits   = 8
  ,parameter   int unsigned InChannels  = 8
  ,parameter   int unsigned Unsigned    = 0
  ,parameter   logic signed [BiasBits-1:0] Bias = '0
  ,localparam  int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0] valid_i
  ,input [0:0] ready_i

  ,input logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_i
  ,input logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_i

  ,output [0:0] valid_o
  ,output [0:0] ready_o
  ,output logic signed [OutBits-1:0] data_o
);

  /* ------------------------------------ Output Channels ------------------------------------ */
  logic signed [InChannels-1:0][AccBits-1:0] kernel_data_d;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_channels
      mac #(
         .InBits    (InBits)
        ,.OutBits   (AccBits)
        ,.WeightBits(WeightBits)
        ,.TermCount (KernelArea)
        ,.Unsigned  (Unsigned)
      ) mac_inst (
         .window_i (windows_i[ch])
        ,.weights_i(weights_i[ch])
        ,.sum_o    (kernel_data_d[ch])
      );
    end
  endgenerate

  /* ----------------------------- Filter Output Summation Logic ----------------------------- */
  logic signed [AccBits-1:0] sum_pre_elastic_d;
  always_comb begin
    sum_pre_elastic_d = AccBits'(0);
    for (int i = 0; i < InChannels; i++) begin
      sum_pre_elastic_d = $signed(sum_pre_elastic_d) + $signed(kernel_data_d[i]);
    end
  end



  /* ----------------------------- Filter Output Summation Logic ----------------------------- */
  logic signed [AccBits-1:0] biased_sum_d;

  logic signed [OutBits-1:0] data_d, data_q;
  assign data_o = data_q;

  assign biased_sum_d = sum_pre_elastic_d + AccBits'($signed(Bias));

  output_encoder #(
     .InBits   (AccBits)
    ,.OutBits  (OutBits)
    ,.ShiftBits(ShiftBits)
  ) out_enc_inst (
     .data_i (biased_sum_d[AccBits-1:0])
    ,.data_o (data_d)
  );

  elastic #(
     .InBits       (OutBits)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) output_elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)

    ,.valid_i(valid_i) 
    ,.ready_o(ready_o) 
    ,.data_i (data_d)

    ,.valid_o(valid_o) // Top level valid_o
    ,.ready_i(ready_i) // Top level ready_i
    ,.data_o (data_q)  // Top level data_o registered
  );


endmodule
