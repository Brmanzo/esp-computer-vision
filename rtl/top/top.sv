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
  ,output uart_rts_o

  ,output [5:1] led_o);

  wire [0:0]  clk_o;

  wire [0:0] reset_n_sync_r;
  wire [0:0] reset_sync_r;
  wire [0:0] reset_r; // Use this as your reset_signal

  wire [3:1] button_sync_r;
  wire [3:1] button_r;

  wire [0:0] rts_w;

  assign uart_rts_o = rts_w;
  assign led_o[1] = rts_w;

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

  // Synchronize and Debounce Buttons
  generate
    for(genvar idx = 1; idx <= 3; idx++) begin
        dff
          #()
        sync_a
          (.clk_i(clk_25mhz_o)
          ,.reset_i(1'b0)
          ,.en_i(1'b1)
          ,.d_i(button_async_unsafe_i[idx])
          ,.q_o(button_sync_r[idx]));

        dff
          #()
        sync_b
          (.clk_i(clk_25mhz_o)
          ,.reset_i(1'b0)
          ,.en_i(1'b1)
          ,.d_i(button_sync_r[idx])
          ,.q_o(button_r[idx]));
    end
  endgenerate
       
  (* blackbox *)
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
       (.clk_i                          (clk_25mhz_o),
        .reset_i                        (reset_r),

        .button_i                       (button_r[3:1]),
        .rx_serial_i                    (ESP_RX_i),
        .tx_serial_o                    (ESP_TX_o),

        .led_o                          (led_o[5:1]),
        .uart_rts_o                     (rts_w));

endmodule
