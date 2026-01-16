`timescale 1ns / 1ps
module sobel
  #(
     parameter linewidth_px_p = 16
    ,parameter in_width_p  = 2
    ,parameter out_width_p = 32
    ,parameter kernel_width_p = 3
    ,parameter weight_width_p = 2)
   (input  [0:0]            clk_i
   ,input  [0:0]            reset_i
   ,input  [0:0]            valid_i
   ,output [0:0]            ready_o
   ,input  [in_width_p-1:0] data_i
   ,output [0:0]            valid_o
   ,input  [0:0]            ready_i
   ,input logic signed [(kernel_width_p*kernel_width_p)-1:0][weight_width_p-1:0] weights_i
   ,output signed [out_width_p-1:0] data_o
   );

   localparam extension_width_lp = out_width_p - in_width_p;


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
   logic [in_width_p-1:0] ram_ol [0:kernel_width_p-1];
   assign ram_ol[0] = data_i;

   generate
      for (genvar i = 0; i < kernel_width_p - 1; i++) begin : GEN_Buffer
         delaybuffer
         #(.width_p(in_width_p)
         ,.delay_p(linewidth_px_p - 1))
         ram_1_delaybuffer_inst
         (.clk_i(clk_i)
         ,.reset_i(reset_i)
         ,.data_i(ram_ol[i])
         ,.valid_i(enable_w)
         ,.ready_o()
         ,.valid_o()
         ,.data_o(ram_ol[i+1])
         ,.ready_i(1'b1)
         );
      end
   endgenerate

   /* ------------------------------------ Window Register Logic ------------------------------------ */
   // Row major Order(window_l[row][col])
   logic [in_width_p-1:0] window_l [kernel_width_p-1:0][kernel_width_p-1:0];

   always_ff @(posedge clk_i) begin
      if (reset_i) begin
         for (int i = 0; i < kernel_width_p; i++) begin
            for (int j = 0; j < kernel_width_p; j++) begin
               window_l[i][j] <= '0;
            end
         end
      end else if (enable_w) begin
         /* ------------------------------- Internal Connections ------------------------------- */ 
         // Line feeds from right to left within window
         // Bottom line
         for (int r = 0; r < kernel_width_p; r++) begin
            for (int c = 0; c < kernel_width_p - 1; c++) begin
               window_l[r][c] <= window_l[r][c+1];
            end
         end
         /* -------------------------------- Input Connections -------------------------------- */ 
         // Load new data into the rightmost column of the window
         // Top line <- Twice seen data from second RAM
         for (int r = 0; r < kernel_width_p; r++) begin
            window_l[r][kernel_width_p-1] <= ram_ol[kernel_width_p-1 - r];
         end
      end
   end
   /* ------------------------------------ Output Logic ------------------------------------ */
   logic signed [out_width_p-1:0] acc_l;
   always_comb begin
      acc_l = '0;
      for (int r = 0; r < kernel_width_p; r++) begin
         for (int c = 0; c < kernel_width_p; c++) begin
            // When binary inputs, only add the weight if the input pixel is a 1
            if (in_width_p == 1) begin
               if (window_l[r][c] != '0) begin
                  acc_l = acc_l + $signed({{extension_width_lp{1'b0}}, weights_i[r*kernel_width_p + c]});
               end
            end else begin
               acc_l = acc_l + ($signed(weights_i[r*kernel_width_p + c]) * $signed(window_l[r][c]));
            end
         end
      end
   end   
   assign data_o = acc_l;

endmodule
