// window.sv
// Bradley Manzo, 2026
`timescale 1ns / 1ps

module window #(
   parameter  int unsigned KernelWidth  = 3
  ,parameter  int unsigned WidthIn      = 1
  ,localparam int unsigned KernelArea   = KernelWidth * KernelWidth
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0] in_fire_i
  ,input [KernelWidth-1:0][WidthIn-1:0] row_buffers_i 

  ,output logic [KernelArea-1:0][WidthIn-1:0] window_o
);
  /* ------------------------------------ Window Register Logic ------------------------------------ */
  // Row major Order(window[row][col])
  logic [KernelArea-1:0][WidthIn-1:0] window_r; // Packed 1D array to hold the kernel window pixels for MAC input

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
        for (int r = 0; r < KernelWidth; r++) begin
          for (int c = 0; c < KernelWidth; c++) begin
              window_r[r*KernelWidth + c] <= '0;
          end
        end
    end else if (in_fire_i) begin
      /* ------------------------------- Internal Connections ------------------------------- */
      // Line feeds from right to left within window
      // Bottom line
      for (int r = 0; r < KernelWidth; r++) begin
        for (int c = 0; c < KernelWidth - 1; c++) begin
            window_r[r*KernelWidth + c] <= window_r[r*KernelWidth + c+1];
        end
      end
      /* -------------------------------- Input Connections -------------------------------- */
      // Load new data into the rightmost column of the window
      // Top line <- Twice seen data from second RAM
      for (int r = 0; r < KernelWidth; r++) begin
        window_r[r*KernelWidth + KernelWidth-1] <= row_buffers_i[KernelWidth-1 - r];
      end
    end
  end

  assign window_o = window_r;

endmodule
