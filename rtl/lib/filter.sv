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
  ,parameter   int unsigned InChannels  = 8
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
    for (int i = 0; i < InChannels; i++) begin : gen_acc_pre_elastic
      sum_pre_elastic_d += kernel_data_d[i];
    end
  end

  /* -------------------------------- Elastic Pipeline Stage -------------------------------- */
  logic signed [AccBits-1:0] sum_pre_elastic_q;
  wire [0:0] elastic_0_valid, elastic_1_ready;
  elastic #(
     .InBits        (AccBits) // Narrower! Only piped sum. Bias is now a parameter.
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)

    ,.valid_i(valid_i)
    ,.ready_o(ready_o)
    ,.data_i (sum_pre_elastic_d)

    ,.valid_o(elastic_0_valid)
    ,.ready_i(elastic_1_ready)
    ,.data_o (sum_pre_elastic_q)
  );

  /* ----------------------------- Filter Output Summation Logic ----------------------------- */
  logic signed [AccBits-1:0] biased_sum_d;

  logic signed [OutBits-1:0] data_d, data_q;
  assign data_o = data_q;

  always_comb begin
    // If binary activation, encode a positive sum as 1 and a negative sum as 0
    biased_sum_d = sum_pre_elastic_q + AccBits'($signed(Bias)); // Use parameter Bias
    if (OutBits == 1) begin
      data_d = (biased_sum_d > 0) ? OutBits'(1) : OutBits'(0);
    // OutBits==2: ternary sign decision matching quantize.py convention (1 if positive, -1 if negative, 0 if zero)
    end else if (OutBits == 2) begin
      data_d = (biased_sum_d > 0) ? OutBits'(1) :
               (biased_sum_d < 0) ? OutBits'(-1) :
                                    OutBits'(0);
    end else if (OutBits == AccBits) begin
      data_d = OutBits'(biased_sum_d);
    end else begin
      // Bit-slicing (Arithmetic Shift Right)
      // Take the top OutBits of the accumulator
      data_d = biased_sum_d[AccBits-1 -: OutBits];
    end
  end

  elastic #(
     .InBits       (OutBits)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) output_elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)

    ,.valid_i(elastic_0_valid) // Comes from the first elastic stage
    ,.ready_o(elastic_1_ready) // Backpressures the first elastic stage
    ,.data_i (data_d)

    ,.valid_o(valid_o) // Top level valid_o
    ,.ready_i(ready_i) // Top level ready_i
    ,.data_o (data_q)  // Top level data_o (now safely registered!)
  );


endmodule
