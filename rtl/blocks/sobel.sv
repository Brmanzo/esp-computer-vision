`timescale 1ns / 1ps
module sobel
  #(
     parameter linewidth_px_p = 16
    ,parameter in_width_p  = 2
    ,parameter out_width_p = 32)
   (input  [0:0]            clk_i
   ,input  [0:0]            reset_i
   ,input  [0:0]            valid_i
   ,output [0:0]            ready_o
   ,input  [in_width_p-1:0] data_i
   ,output [0:0]            valid_o
   ,input  [0:0]            ready_i
   ,input logic signed [8:0][1:0] weights_i
   ,output signed [out_width_p-1:0] data_o
   );

   logic [in_width_p-1:0] window_l [2:0][2:0];
   logic [in_width_p-1:0] ram_1_ol, ram_2_ol;

   /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
   // Provided Elastic State Machine Logic
   logic [0:0] valid_r;
   wire  [0:0] enable_w;
   
   assign enable_w = valid_i & ready_o;
   assign ready_o = ~valid_o | ready_i;

   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         valid_r <= 1'b0;
      end else if (ready_o) begin
         valid_r <= enable_w;
      end
   end
   assign valid_o = valid_r;

   /* ------------------------------------ FIFO RAM Instantiations ------------------------------------ */
   // Both RAMs step forward with each new pixel input (enable_w) so that delay is exactly one row each.
   // RAM 1 produces a delay of one row. Reads off of data_i as the bottom row of the kernel receives new data.
   /* verilator lint_off PINCONNECTEMPTY */
   delaybuffer
   #(.width_p(in_width_p)
   ,.delay_p(linewidth_px_p - 1))
   ram_1_delaybuffer_inst
   (.clk_i(clk_i)
   ,.reset_i(reset_i)
   ,.data_i(data_i)
   ,.valid_i(enable_w)
   ,.ready_o()
   ,.valid_o()
   ,.data_o(ram_1_ol)
   ,.ready_i(1'b1)
   );
   // RAM 2 produces a delay of two rows. Reads off of output of RAM 1 as the middle row receives new data.
   delaybuffer
   #(.width_p(in_width_p)
   ,.delay_p(linewidth_px_p - 1))
   ram_2_delaybuffer_inst
   (.clk_i(clk_i)
   ,.reset_i(reset_i)
   ,.data_i(ram_1_ol)
   ,.valid_i(enable_w)
   ,.ready_o()
   ,.valid_o()
   ,.data_o(ram_2_ol)
   ,.ready_i(1'b1));
   /* ------------------------------------ Window Register Logic ------------------------------------ */
   // Row major Order(window_l[row][col])
   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         for (int i = 0; i < 3; i++) begin
            for (int j = 0; j < 3; j++) begin
               window_l[i][j] <= '0;
            end
         end
      end else if (enable_w) begin
         /* ------------------------------- Internal Connections ------------------------------- */ 
         // Line feeds from right to left within window
         // Bottom line
         window_l[2][1] <= window_l[2][2];
         window_l[2][0] <= window_l[2][1];

         // Middle line
         window_l[1][1] <= window_l[1][2];
         window_l[1][0] <= window_l[1][1];

         // Top line
         window_l[0][1] <= window_l[0][2];
         window_l[0][0] <= window_l[0][1];
         /* -------------------------------- Input Connections -------------------------------- */ 
         // Load new data into the rightmost column of the window
         // Bottom line <- Newest data from input
         window_l[2][2] <= data_i;
         // Middle line <- Once seen data from first RAM
         window_l[1][2] <= ram_1_ol;
         // Top line <- Twice seen data from second RAM
         window_l[0][2] <= ram_2_ol;
      end
   end
   /* ------------------------------------ Output Logic ------------------------------------ */
   logic signed [out_width_p-1:0] acc_l;
   logic signed [out_width_p-1:0] pp_l;
   always_comb begin
      acc_l = '0;
      for (int r = 0; r < 3; r++) begin
         for (int c = 0; c < 3; c++) begin
            // When binary inputs, use bitwise AND instead of multiplication
            if (in_width_p == 1) begin
               pp_l = signed'(out_width_p'($signed(weights_i[r*3 + c])) & out_width_p'(window_l[r][c]));
               acc_l = acc_l + pp_l;
            end else begin
               acc_l = acc_l + ($signed(weights_i[r*3 + c]) * $signed(window_l[r][c]));
            end
         end
      end
   end   
   assign data_o = acc_l;

endmodule
