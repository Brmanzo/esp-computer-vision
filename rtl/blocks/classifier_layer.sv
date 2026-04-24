// classifier_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module classifier_layer #(
   parameter  int unsigned TermBits   = 8 // Preserved between global max and comparator tree to maintain precision for comparisons
  ,parameter  int unsigned TermCount  = 32
  ,parameter  int unsigned BusBits    = 8 // Output bus width
  ,parameter  int unsigned InChannels = 1
  ,parameter  int unsigned ClassCount = 1
  ,localparam int unsigned IdBits     = (ClassCount <= 1) ? 1 : $clog2(ClassCount)

  ,parameter  int unsigned WeightBits = 2
  ,parameter  int unsigned BiasBits   = 4

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [ClassCount*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [ClassCount*BiasBits-1:0]    Biases  = '0
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o

  ,input  signed [InChannels-1:0][TermBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output [BusBits-1:0] class_o
);

  function automatic int unsigned acc_width;
    input int unsigned input_width, weight_width, in_channels;
    longint unsigned max_input, max_weight, worst_case_sum;
    begin
      max_input      = (64'd1 << input_width) - 1;
      max_weight     = (64'd1 << (weight_width - 1)) - 1;
      worst_case_sum = max_input * max_weight * longint'(in_channels);

      acc_width = $clog2(worst_case_sum + 1) + 1; // assign to function name
    end
  endfunction

  localparam int unsigned LinearBits = acc_width(TermBits, WeightBits, InChannels);

  wire [0:0] global_max_valid;
  wire signed [InChannels-1:0][TermBits-1:0] global_max_data;

  wire [0:0] linear_layer_ready;
  wire signed [ClassCount-1:0][LinearBits-1:0] linear_layer_data;

  wire [IdBits-1:0] id_o;

  global_max #(
     .InBits     (TermBits)
    ,.TermCount  (TermCount)
    ,.InChannels (InChannels)
  ) global_max_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.valid_i (valid_i)
    ,.ready_o (ready_o)
    ,.data_i  (data_i)

    ,.ready_i (linear_layer_ready)
    ,.valid_o (global_max_valid)
    ,.data_o  (global_max_data)
  );

  linear_layer #(
     .InBits      (TermBits)
    ,.OutBits     (LinearBits)
    ,.WeightBits  (WeightBits)
    ,.BiasBits    (BiasBits)
    ,.InChannels  (InChannels)
    ,.OutChannels (ClassCount)
    ,.Weights     (Weights)
    ,.Biases      (Biases)
  ) linear_layer_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o (linear_layer_ready)
    ,.valid_i (global_max_valid)
    ,.data_i  (global_max_data)

    ,.ready_i (ready_i)
    ,.valid_o (valid_o)
    ,.data_o  (linear_layer_data)
  );

  /* verilator lint_off PINCONNECTEMPTY */
  comparator_tree #(
     .InBits     (LinearBits)
    ,.ClassCount (ClassCount)
  ) comparator_tree_inst (
     .classes_i (linear_layer_data)
    ,.max_o     ()
    ,.id_o      (id_o) // Outputs minimum bit width to encode class count
  );

  // Extend class ID to the designated output width of bus
  assign class_o = BusBits'(id_o);

endmodule
