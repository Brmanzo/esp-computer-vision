// test_cnn_full.sv
// Wrapper for the full CNN integration test

`timescale 1ns / 1ps

module cnn_full #(
  parameter int unsigned BusBits = 8
) (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,input  [0:0] ready_i
  ,input  [0:0] data_i

  ,output [0:0] valid_o
  ,output [0:0] ready_o
  ,output [BusBits-1:0] data_o
);

  // Instantiate the auto-generated CNN top-level
  cnn #(
    .BusBits(BusBits)
  ) dut (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.valid_i (valid_i)
    ,.ready_i (ready_i)
    ,.data_i  (data_i)
    ,.valid_o (valid_o)
    ,.ready_o (ready_o)
    ,.data_o  (data_o)
  );

  // Add waveform dumping for debugging
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("cnn_full.fst");
    $dumpvars(0, cnn_full);
  end
`endif

endmodule
