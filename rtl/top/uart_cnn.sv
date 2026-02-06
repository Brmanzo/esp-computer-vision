// uart_cnn.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
/* verilator lint_off PINCONNECTEMPTY */
module uart_cnn #(
   parameter int unsigned WidthIn        = 320
  ,parameter int unsigned HeightIn       = 240
  ,parameter int unsigned KernelWidth    = 3
  ,parameter int unsigned WeightWidth    = 2
  ,parameter int unsigned BusWidth       = 8
  ,parameter int unsigned QuantizedWidth = 1
  ,parameter int unsigned PackedNum      = BusWidth / QuantizedWidth
  ,parameter int unsigned Channels       = 2

  ,localparam int unsigned BytesIn       = WidthIn * HeightIn / PackedNum
  ,localparam int unsigned KernelArea    = KernelWidth * KernelWidth
  ,localparam int unsigned MaxInput      = (1 << QuantizedWidth) - 1
  ,localparam int unsigned MaxWeight     = (1 << (WeightWidth - 1)) - 1
  ,localparam int unsigned WorstCaseSum  = KernelArea * MaxInput * MaxWeight
  ,localparam int unsigned ConvOutWidth  = $clog2(WorstCaseSum + 1) + 1 // Plus signed bit
  ,localparam int unsigned MagOutWidth   = ConvOutWidth + 1 // Plus bit for magnitude potentially being larger than either Gx or Gy

  ,localparam int unsigned WidthOut  = WidthIn - (KernelWidth - 1)
  ,localparam int unsigned HeightOut = HeightIn - (KernelWidth - 1)

  ,localparam int unsigned ConvThreshold = 2
  ,localparam int unsigned MagThreshold  = ConvThreshold << 1
  ,localparam logic [ConvOutWidth-1:0] ConvThresh = ConvOutWidth'(ConvThreshold)
  ,localparam logic [MagOutWidth-1:0]  MagThresh  = MagOutWidth'(MagThreshold)
)  (
   input  [0:0] clk_i // 25 MHz Clock
  ,input  [0:0] rst_i
  ,input  [3:1] button_i

  ,input  [0:0] rx_serial_i
  ,output [0:0] tx_serial_o
  ,output [0:0] uart_rts_o

  ,output [5:1] led_o
);

  // UART Interface Wires
  wire [0:0]                uart_ready;
  wire [0:0]                uart_valid;
  wire [BusWidth-1:0]       uart_data;

  // Skid Buffer Wires
  wire [0:0]                skid_ready;
  wire [0:0]                skid_valid;
  wire [BusWidth-1:0]       skid_data;

  // Deframer Wires
  wire [0:0]                deframer_ready;
  wire [0:0]                deframer_valid;
  wire [QuantizedWidth-1:0] deframer_data;

  // conv_layer Wires
  wire [0:0]                               conv_layer_ready;
  wire [0:0]                               conv_layer_valid;
  wire signed [Channels-1:0][ConvOutWidth-1:0] conv_layer_data;

  // Magnitude Wires
  wire [0:0]              mag_ready;
  wire [0:0]              mag_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  wire [MagOutWidth-1:0]  mag_data;
  /* verilator lint_on UNUSEDSIGNAL */

  // Mux Logic
  logic [0:0]             mux_data;
  logic [0:0]             mux_valid;

  // Framer Wires
  wire [0:0]              framer_ready;
  wire [0:0]              framer_valid;
  wire [BusWidth-1:0]     framer_data;

  assign led_o[5:1] = 5'b0;

  // Predefined Kernel weights for gx and gy gradients
  typedef logic signed [WeightWidth-1:0] weight_t;

  function automatic weight_t gx(input int unsigned i);
    unique case (i)
      0: gx = 2'sd1;  1: gx = 2'sd0;  2: gx = -2'sd1;
      3: gx = 2'sd1;  4: gx = 2'sd0;  5: gx = -2'sd1;
      6: gx = 2'sd1;  7: gx = 2'sd0;  8: gx = -2'sd1;
      default: gx = '0;
    endcase
  endfunction

  function automatic weight_t gy(input int unsigned i);
    unique case (i)
      0: gy =  2'sd1;  1: gy =  2'sd1;  2: gy =  2'sd1; 
      3: gy =  2'sd0;  4: gy =  2'sd0;  5: gy =  2'sd0;
      6: gy = -2'sd1;  7: gy = -2'sd1;  8: gy = -2'sd1;
      default: gy = '0;
    endcase
  endfunction

  logic signed [Channels-1:0][KernelArea-1:0][WeightWidth-1:0] weights;
  genvar k;
  generate
    for (k = 0; k < KernelArea; k++) begin : gen_gy
      assign weights[0][k] = gx(k);  // gx_w returns weight_t (signed [2:0])
      assign weights[1][k] = gy(k);  // gy_w returns weight_t (signed [2:0])
    end
  endgenerate

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
    ,.s_axis_tvalid(framer_valid)
    ,.s_axis_tdata (framer_data)
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
     .Width   (BusWidth)
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
     .UnpackedWidth(QuantizedWidth)
    ,.PackedNum(PackedNum)
    ,.PacketLenElems(BytesIn)
    ,.HeaderByte0(8'hA5)
    ,.HeaderByte1(8'h5A)
  ) deframer_inst (
     .clk_i(clk_i)
    ,.rst_i(rst_i)

    ,.valid_i(skid_valid)
    ,.ready_o(deframer_ready)
    ,.data_i(skid_data)

    ,.valid_o(deframer_valid)
    ,.ready_i(conv_layer_ready)
    ,.unpacked_o(deframer_data)
  );

  // conv_layer Filter for Gy gradient
  conv_layer #(
     .LineWidthPx(WidthIn)
    ,.LineCountPx(HeightIn)
    ,.WidthIn    (QuantizedWidth)
    ,.WidthOut   (ConvOutWidth)
    ,.KernelWidth(KernelWidth)
    ,.WeightWidth(WeightWidth)
    ,.Channels   (Channels)
  ) conv_layer_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    // Unpacker to Gy
    ,.ready_o  (conv_layer_ready)
    ,.valid_i  (deframer_valid)
    ,.data_i   (deframer_data)
    // Gy to Elastic Stage
    ,.ready_i  (mag_ready)
    ,.valid_o  (conv_layer_valid)
    ,.data_o   (conv_layer_data)
    ,.weights_i(weights)
  );

  // Magnitude calculated from Gx and Gy
  mag #(
     .Width  (ConvOutWidth)
  ) mag_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    // Elastic Stage to Magnitude
    ,.ready_o(mag_ready)
    ,.valid_i(conv_layer_valid)
    ,.gx_i   (conv_layer_data[0])
    ,.gy_i   (conv_layer_data[1])
    // Magnitude to Packer
    ,.ready_i(framer_ready)
    ,.valid_o(mag_valid)
    ,.mag_o  (mag_data)
  );

  // Instead of selecting arbitrary bit off of signed conv output, take absolute value above a given
  logic gx_edge  = ($signed(conv_layer_data[0]) >= $signed(ConvThresh));
  logic gy_edge  = ($signed(conv_layer_data[1]) >= $signed(ConvThresh));
  logic mag_edge = (mag_data >= MagThresh);

  always_comb begin
    case (button_i)
      3'b001:  begin mux_data = deframer_data; mux_valid = deframer_valid; end
      3'b010:  begin mux_data = gx_edge;       mux_valid = mag_valid;      end
      3'b100:  begin mux_data = gy_edge;       mux_valid = mag_valid;      end
      default: begin mux_data = mag_edge;      mux_valid = mag_valid;      end
    endcase
  end
  // Packs 8 pixels onto a single byte, adds footer at end of frame and sends to UART
  framer #(
     .UnpackedWidth (QuantizedWidth)
    ,.PackedNum     (PackedNum)
    ,.PacketLenElems(WidthOut * HeightOut)
    ,.TailByte0     (8'hA5)
    ,.TailByte1     (8'h5A)
  ) framer_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    // Mux to Framer
    ,.unpacked_i(mux_data)
    ,.valid_i   (mux_valid)
    ,.ready_o   (framer_ready)
    // Framer to Packer
    ,.data_o    (framer_data)
    ,.valid_o   (framer_valid)
    ,.ready_i   (uart_ready)
  );

endmodule
