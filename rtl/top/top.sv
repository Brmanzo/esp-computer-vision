// top-level design file for the icebreaker FPGA board
module top
  (input [0:0] clk_12mhz_i
  // n: Negative Polarity (0 when pressed, 1 otherwise)
  // async: Not synchronized to clock
  // unsafe: Not De-Bounced
  ,input [0:0] reset_n_async_unsafe_i
  // async: Not synchronized to clock
  // unsafe: Not De-Bounced
  ,input [3:1] button_async_unsafe_i

  // Serial Interface
  ,input rx_serial_i
  // Input data
  ,input  ESP_RX_i
  ,output ESP_TX_o
  ,output tx_serial_o

  ,output [5:1] led_o);

   wire        clk_o;

   // These two D Flip Flops form what is known as a Synchronizer. We
   // will learn about these in Week 5, but you can see more here:
   // https://inst.eecs.berkeley.edu/~cs150/sp12/agenda/lec/lec16-synch.pdf
   wire reset_n_sync_r;
   wire reset_sync_r;
   wire reset_r; // Use this as your reset_signal

   dff
     #()
   sync_a
     (.clk_i(clk_25mhz_o)
     ,.reset_i(1'b0)
     ,.en_i(1'b1)
     ,.d_i(reset_n_async_unsafe_i)
     ,.q_o(reset_n_sync_r));

   inv
     #()
   inv
     (.a_i(reset_n_sync_r)
     ,.b_o(reset_sync_r));

   dff
     #()
   sync_b
     (.clk_i(clk_25mhz_o)
     ,.reset_i(1'b0)
     ,.en_i(1'b1)
     ,.d_i(reset_sync_r)
     ,.q_o(reset_r));
       
  (* blackbox *)
  // This is a PLL! You'll learn about these later...
  SB_PLL40_2_PAD
    #(.FEEDBACK_PATH("SIMPLE")
     ,.DIVR(4'b0000)
     ,.DIVF(7'd66)
     ,.DIVQ(3'd5)
     ,.FILTER_RANGE(3'b001)
     )
   pll_inst
     (.PACKAGEPIN(clk_12mhz_i)
     ,.PLLOUTGLOBALA(clk_12mhz_o)
     ,.PLLOUTGLOBALB(clk_25mhz_o)
     ,.RESETB(1'b1)
     ,.BYPASS(1'b0)
     );
  

   uart_axis
     uart_axis_i
       (.clk_i                          (clk_25mhz_o), // 25 MHz Clock
        .reset_i                        (reset_r),

        .rx_serial_i                    (rx_serial_i),
        .tx_serial_o                    (tx_serial_o),

        .led_o                          (led_o[5:1]));

endmodule
