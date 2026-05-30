// test_cnn_full.sv
// Wrapper for the full CNN integration test

`timescale 1ns / 1ps

module cnn_framed #(
  parameter int unsigned BusBits = 8
  ,parameter              FileName_0    = "nn/data/roms/hex/zeros.hex"
  ,parameter              FileName_1    = "nn/data/roms/hex/zeros.hex"
  ,parameter              FileName_2    = "nn/data/roms/hex/zeros.hex"
  ,parameter              FileName_3    = "nn/data/roms/hex/zeros.hex"
  ,parameter              FileName_4    = "nn/data/roms/hex/zeros.hex"
  ,parameter              FileName_5    = "nn/data/roms/hex/zeros.hex"

  ,parameter  int unsigned WidthIn   = 320
  ,parameter  int unsigned HeightIn  = 240
  ,localparam int unsigned InBits    = 1
  ,localparam int unsigned PackedNum = BusBits / InBits
  ,localparam int unsigned BytesIn   = (WidthIn * HeightIn) / PackedNum
) (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,input  [0:0] ready_i
  ,input  [BusBits-1:0] data_i

  ,output [0:0] valid_o
  ,output [0:0] ready_o
  ,output [BusBits-1:0] data_o
);

  wire [0:0]        deframer_valid;
  wire [InBits-1:0] deframer_data;

  wire [0:0]         cnn_ready;
  wire [0:0]         cnn_valid;
  wire [BusBits-1:0] cnn_data;

  wire [0:0]        class_framer_ready;

  deframer #(
     .UnpackedWidth (InBits)
    ,.PackedNum     (PackedNum)
    ,.PacketLenElems(BytesIn)
  ) deframer_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o (ready_o)
    ,.valid_i (valid_i)
    ,.data_i  (data_i)

    ,.ready_i    (cnn_ready)
    ,.valid_o    (deframer_valid)
    ,.unpacked_o (deframer_data)
  );

  // Instantiate the auto-generated CNN top-level
  cnn #(
     .BusBits      (BusBits)
    ,.FileName_0   (FileName_0)
    ,.FileName_1   (FileName_1)
    ,.FileName_2   (FileName_2)
    ,.FileName_3   (FileName_3)
    ,.FileName_4   (FileName_4)
    ,.FileName_5   (FileName_5)
  ) dut (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.ready_o (cnn_ready)
    ,.valid_i (deframer_valid)
    ,.data_i  (deframer_data)
    
    ,.ready_i (class_framer_ready)
    ,.valid_o (cnn_valid)
    ,.data_o  (cnn_data)
  );

  class_framer #(
      .BusBits   (BusBits)
    ) framer_inst (
      .clk_i     (clk_i)
      ,.rst_i    (rst_i)
  
      ,.ready_o (class_framer_ready)
      ,.class_i (cnn_data)
      ,.valid_i (cnn_valid)
  
      ,.ready_i (ready_i)
      ,.valid_o (valid_o)
      ,.uart_o  (data_o)
  );

  // Add waveform dumping for debugging
`ifdef COCOTB_SIM
  initial begin
    $dumpfile("cnn_framed.fst");
    $dumpvars(0, cnn_framed);
  end
`endif

endmodule
