`timescale 1ns / 1ps
/* verilator lint_off PINCONNECTEMPTY */
module uart_axis #(
   parameter int unsigned ImageWidth     = 320
  ,parameter int unsigned ImageHeight    = 240
  ,parameter int unsigned KernelWidth    = 3
  ,parameter int unsigned WeightWidth    = 2
  ,parameter int unsigned BusWidth       = 8
  ,parameter int unsigned QuantizedWidth = 1
  ,parameter int unsigned PackedNum      = BusWidth / QuantizedWidth

  ,localparam int unsigned KernelArea   = KernelWidth * KernelWidth
  ,localparam int unsigned MaxInput     = (1 << QuantizedWidth) - 1
  ,localparam int unsigned MaxWeight    = (1 << (WeightWidth - 1)) - 1
  ,localparam int unsigned WorstCaseSum = KernelArea * MaxInput * MaxWeight
  ,localparam int unsigned ConvOutWidth = $clog2(WorstCaseSum + 1) + 1 // Plus signed bit

  ,localparam int unsigned WidthOut  = ImageWidth - (KernelWidth - 1)
  ,localparam int unsigned HeightOut = ImageHeight - (KernelWidth - 1)
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

  // Unpacker Wires
  wire [0:0]                unpack_ready;
  wire [0:0]                unpack_valid;
  wire [QuantizedWidth-1:0] unpacked_data;

  // conv2d Wires
  wire [0:0]              gx_ready, gy_ready;
  wire [0:0]              gx_valid, gy_valid;
  wire [ConvOutWidth-1:0] gx_data, gy_data;

  // Magnitude Wires
  wire [0:0]              mag_ready;
  wire [0:0]              mag_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  wire [ConvOutWidth:0]   mag_data;
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

  logic signed [KernelArea-1:0][WeightWidth-1:0] gx_weights;

  genvar j;
  generate
    for (j = 0; j < KernelArea; j++) begin : gen_gx
      assign gx_weights[j] = gx(j);  // gx returns weight_t (signed [2:0])
    end
  endgenerate

  function automatic weight_t gy(input int unsigned i);
    unique case (i)
      0: gy =  2'sd1;  1: gy =  2'sd1;  2: gy =  2'sd1; 
      3: gy =  2'sd0;  4: gy =  2'sd0;  5: gy =  2'sd0;
      6: gy = -2'sd1;  7: gy = -2'sd1;  8: gy = -2'sd1;
      default: gy = '0;
    endcase
  endfunction

  logic signed [KernelArea-1:0][WeightWidth-1:0] gy_weights;

  genvar k;
  generate
    for (k = 0; k < KernelArea; k++) begin : gen_gy
      assign gy_weights[k] = gy(k);  // gy_w returns weight_t (signed [2:0])
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
    ,.ready_i(unpack_ready)
    ,.rts_o  (uart_rts_o)
  );

  // Unpacker to unpack 4 2-bit values from each 8-bit UART input
  unpacker #(
     .UnpackedWidth(QuantizedWidth)
    ,.PackedNum    (PackedNum)
  ) unpacker_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    // Skid Buffer to Unpacker
    ,.ready_o   (unpack_ready)
    ,.valid_i   (skid_valid)
    ,.packed_i  (skid_data)
    // Unpacker to conv2d Filters
    ,.ready_i   (gx_ready & gy_ready)
    ,.valid_o   (unpack_valid)
    ,.unpacked_o(unpacked_data)
  );

  // conv2d Filter for Gx gradient
  conv2d #(
     .LineWidthPx(ImageWidth)
    ,.LineCountPx(ImageHeight)
    ,.WidthIn    (QuantizedWidth + 1) // Zero pad inputs
    ,.WidthOut   (ConvOutWidth)
    ,.KernelWidth(KernelWidth)
    ,.WeightWidth(WeightWidth)
  ) conv2d_gx_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    // Unpacker to Gx
    ,.ready_o  (gx_ready)
    ,.valid_i  (unpack_valid)
    ,.data_i   ({1'b0, unpacked_data}) // "Right shift" by 3 to divide by 8 and average the output
    // Gx to Elastic Stage
    ,.ready_i  (mag_ready)
    ,.valid_o  (gx_valid)
    ,.data_o   (gx_data)
    ,.weights_i(gx_weights)
  );

  // conv2d Filter for Gy gradient
  conv2d #(
     .LineWidthPx(ImageWidth)
    ,.LineCountPx(ImageHeight)
    ,.WidthIn    (QuantizedWidth + 1)
    ,.WidthOut   (ConvOutWidth)
    ,.KernelWidth(KernelWidth)
    ,.WeightWidth(WeightWidth)
  ) conv2d_gy_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    // Unpacker to Gy
    ,.ready_o  (gy_ready)
    ,.valid_i  (unpack_valid)
    ,.data_i   ({1'b0, unpacked_data})
    // Gy to Elastic Stage
    ,.ready_i  (mag_ready)
    ,.valid_o  (gy_valid)
    ,.data_o   (gy_data)
    ,.weights_i(gy_weights)
  );

  // Magnitude calculated from Gx and Gy
  mag #(
     .Width  (ConvOutWidth)
  ) mag_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    // Elastic Stage to Magnitude
    ,.ready_o(mag_ready)
    ,.valid_i(gx_valid & gy_valid)
    ,.gx_i   (gx_data)
    ,.gy_i   (gy_data)
    // Magnitude to Packer
    ,.ready_i(framer_ready)
    ,.valid_o(mag_valid)
    ,.mag_o  (mag_data)
  );

  always_comb begin
    case (button_i)
      3'b001:  begin mux_data = unpacked_data; mux_valid = unpack_valid; end
      3'b010:  begin mux_data = gx_data[2];    mux_valid = mag_valid;    end
      3'b100:  begin mux_data = gy_data[2];    mux_valid = mag_valid;    end
      default: begin mux_data = mag_data[2];   mux_valid = mag_valid;    end
    endcase
  end
  // Packs 8 pixels onto a single byte, adds footer at end of frame and sends to UART
  framer #(
     .UnpackedWidth (QuantizedWidth)
    ,.PackedNum     (PackedNum)
    ,.PacketLenElems(HeightOut * WidthOut)
    ,.TailByte0     (8'h0D)
    ,.TailByte1     (8'h0A)
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
