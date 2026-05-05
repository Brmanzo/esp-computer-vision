// uart_cnn.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
/* verilator lint_off PINCONNECTEMPTY */
module uart_cnn #(
   parameter  int unsigned WidthIn   = 320
  ,parameter  int unsigned HeightIn  = 240
  ,parameter  int unsigned BusBits  = 8
  ,parameter  int unsigned InBits    = 1

  ,localparam int unsigned PackedNum = BusBits / InBits
  ,localparam int unsigned BytesIn   = (WidthIn * HeightIn) / PackedNum
)  (
   input  [0:0] clk_i // 25 MHz Clock
  ,input  [0:0] rst_i

  ,input  [0:0] rx_serial_i

  ,output [0:0] tx_serial_o
  ,output [0:0] uart_rts_o
);

  // UART Interface Wires
  wire [0:0]                uart_ready;
  wire [0:0]                uart_valid;
  wire [BusBits-1:0]       uart_data;

  // Skid Buffer Wires
  wire [0:0]                skid_ready;
  wire [0:0]                skid_valid;
  wire [BusBits-1:0]       skid_data;

  // Deframer Wires
  wire [0:0]                deframer_ready;
  wire [0:0]                deframer_valid;
  wire [InBits-1:0] deframer_data;

  // cnn Wires
  wire [0:0]                cnn_ready;
  wire [0:0]                cnn_valid;
  wire [BusBits-1:0]       cnn_data;

  // Class Framer Wires
  wire [0:0]              class_framer_ready;
  wire [0:0]              class_framer_valid;
  wire [BusBits-1:0]     class_framer_data;

  // UART head to convert UART serial data to AXIS data
  uart #(
  ) uart_inst (
     .clk(clk_i)
    ,.rst(rst_i)
    // FPGA interface for UART
    ,.txd(tx_serial_o) // ESP_TX_o pin 2
    ,.rxd(rx_serial_i) // ESP_RX_i pin 4

    // Packer to UART
    ,.s_axis_tready(uart_ready)
    ,.s_axis_tvalid(class_framer_valid)
    ,.s_axis_tdata (class_framer_data)
    // UART to AXIS
    ,.m_axis_tready(skid_ready)
    ,.m_axis_tvalid(uart_valid)
    ,.m_axis_tdata (uart_data)

    ,.tx_busy         ()
    ,.rx_busy         ()
    ,.rx_overrun_error()
    ,.rx_frame_error  ()
    ,.prescale        (16'd10) // Fclk / (baud * 8), 25 MHz / (312,500 * 8) = 10
  );

  skid_buffer #(
     .Width   (BusBits)
    ,.Depth   (16)
    ,.HeadRoom(6)
  ) skid_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    // UART to Skid Buffer
    ,.data_i (uart_data)
    ,.valid_i(uart_valid)
    ,.ready_o(skid_ready)
    // Skid Buffer to Unpacker
    ,.data_o (skid_data)
    ,.valid_o(skid_valid)
    ,.ready_i(deframer_ready)
    ,.rts_o  (uart_rts_o)
  );

  deframer #(
     .UnpackedWidth(InBits)
    ,.PackedNum(PackedNum)
    ,.PacketLenElems(BytesIn)
    ,.HeaderByte0(8'hA5)
    ,.HeaderByte1(8'h5A)
  ) deframer_inst (
     .clk_i(clk_i)
    ,.rst_i(rst_i)
    // Skid Buffer to Deframer
    ,.valid_i(skid_valid)
    ,.ready_o(deframer_ready)
    ,.data_i(skid_data)
    // Deframer to CNN
    ,.ready_i(cnn_ready)
    ,.valid_o(deframer_valid)
    ,.unpacked_o(deframer_data)
  );

  cnn #(
     .BusBits(BusBits)
  ) cnn_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    // Deframer to CNN
    ,.ready_o(cnn_ready)
    ,.valid_i(deframer_valid)
    ,.data_i (deframer_data)
    // CNN to Class Framer
    ,.ready_i(class_framer_ready)
    ,.valid_o(cnn_valid)
    ,.data_o (cnn_data)
  );
  
  class_framer #(
     .BusBits   (BusBits)
    ,.TailByte0 (8'hA5)
    ,.TailByte1 (8'h5A)
  ) framer_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    // CNN to Class Framer
    ,.class_i(cnn_data)
    ,.valid_i(cnn_valid)
    ,.ready_o(class_framer_ready)
    // Class Framer to UART
    ,.uart_o(class_framer_data)
    ,.valid_o(class_framer_valid)
    ,.ready_i(uart_ready)
  );

endmodule
