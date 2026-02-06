// top.sv
// Bradley Manzo, 2026

`define ESP // Comment out this line to target USB-to-UART bridge instead of ESP32c3
module top #(
)  (
   input [0:0] clk_12mhz_i
  ,input [0:0] reset_n_async_unsafe_i
  ,input [3:1] button_async_unsafe_i

  ,input  [0:0] rx_serial_i
  ,output [0:0] tx_serial_o

  ,input  [0:0] esp_rx_i
  ,output [0:0] esp_tx_o

  ,output [0:0] uart_rts_o
  ,output [5:1] led_o
);

  wire [0:0] clk_12mhz_o;
  wire [0:0] clk_25mhz_o;

  wire [0:0] reset_n_sync_q;
  wire [0:0] reset_sync_q;
  wire [0:0] rst; // Use this as your reset_signal

  wire [3:1] button_sync_q;
  wire [3:1] button_q;

  wire [0:0] rts; // RTS to backpressure ESP UART
  assign uart_rts_o = rts;

  dff #(
  ) sync_a (
     .clk_i  (clk_25mhz_o)
    ,.reset_i(1'b0)
    ,.en_i   (1'b1)
    ,.d_i    (reset_n_async_unsafe_i)
    ,.q_o    (reset_n_sync_q)
  );

  inv #(
  ) inv (
     .a_i(reset_n_sync_q)
    ,.b_o(reset_sync_q)
  );

  dff #(
  ) sync_b (
     .clk_i  (clk_25mhz_o)
    ,.reset_i(1'b0)
    ,.en_i   (1'b1)
    ,.d_i    (reset_sync_q)
    ,.q_o    (rst)
  );

  // Synchronize and Debounce Buttons
  generate
    for(genvar idx = 1; idx <= 3; idx++) begin : gen_sync
      dff #(
      ) sync_a ( 
         .clk_i  (clk_25mhz_o)
        ,.reset_i(1'b0)
        ,.en_i   (1'b1)
        ,.d_i    (button_async_unsafe_i[idx])
        ,.q_o    (button_sync_q[idx])
      );

      dff #(
      ) sync_b (
         .clk_i  (clk_25mhz_o)
        ,.reset_i(1'b0)
        ,.en_i   (1'b1)
        ,.d_i    (button_sync_q[idx])
        ,.q_o    (button_q[idx])
      );
    end
  endgenerate
       
  (* blackbox *)
  SB_PLL40_2_PAD #(
     .FEEDBACK_PATH("SIMPLE")
    ,.DIVR        (4'b0000)
    ,.DIVF        (7'd66)
    ,.DIVQ        (3'd5)
    ,.FILTER_RANGE(3'b001)
  ) pll_inst (
     .PACKAGEPIN   (clk_12mhz_i)
    ,.PLLOUTGLOBALA(clk_12mhz_o)
    ,.PLLOUTGLOBALB(clk_25mhz_o)
    ,.RESETB       (1'b1)
    ,.BYPASS       (1'b0)
  );
  

wire [0:0] uart_rx;
wire [0:0] uart_tx;

// Either synthesize to GPIO for ESP32c3
`ifdef ESP
  assign uart_rx = esp_rx_i;
  assign uart_tx = esp_tx_o;

  assign tx_serial_o = 1'b1;
// or synthesize to USB-to-UART bridge for python demo
`else
  assign uart_rx = rx_serial_i;
  assign uart_tx = tx_serial_o;

  assign esp_tx_o = 1'b1;
`endif

  uart_axis #(
     .WidthIn        (320)
    ,.HeightIn       (240)
    ,.KernelWidth    (3)
    ,.WeightWidth    (2)
    ,.BusWidth       (8)
    ,.QuantizedWidth (1)
  ) uart_axis_inst (
     .clk_i      (clk_25mhz_o)
    ,.rst_i      (rst)

    ,.button_i   (button_q[3:1])
    ,.rx_serial_i(uart_rx)
    ,.tx_serial_o(uart_tx)

    ,.led_o      (led_o[5:1])
    ,.uart_rts_o (rts)
  );

endmodule
