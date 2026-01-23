`timescale 1ns / 1ps
module conv2d #(
   parameter  int unsigned LineWidthPx = 160
  ,parameter  int unsigned LineCountPx = 120
  ,parameter  int unsigned WidthIn     = 2
  ,parameter  int unsigned WidthOut    = 32
  ,parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WeightWidth = 2
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
  ,localparam int unsigned ExtendWidth = WidthOut - WidthIn
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0]         valid_i
  ,output [0:0]         ready_o
  ,input  [WidthIn-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [WidthOut-1:0] data_o
  ,input  logic signed [KernelArea-1:0][WeightWidth-1:0] weights_i
);

  /* ---------------------------------------- Kernel Validation ---------------------------------------- */
  wire [0:0] in_fire = valid_i & ready_o;

  localparam int XWidth = (LineWidthPx <= 1) ? 1 : $clog2(LineWidthPx);
  localparam int YWidth = (LineCountPx <= 1) ? 1 : $clog2(LineCountPx);

  logic [XWidth-1:0] x_pos;
  logic [YWidth-1:0] y_pos;

  wire [0:0] last_col = (x_pos == XWidth'(LineWidthPx - 1));
  wire [0:0] last_row = (y_pos == YWidth'(LineCountPx - 1));

  always_ff @(posedge clk_i) begin
    // Update x and y position counters
    if (rst_i) begin
      x_pos <= '0;
      y_pos <= '0;
    end else if (in_fire) begin
      if (last_col) begin
        x_pos <= '0;
        y_pos <= (last_row) ? '0 : (y_pos + 1);
      end else begin
        x_pos <= x_pos + 1;
      end
    end
  end

  wire [0:0] valid_x_pos = (x_pos >= (XWidth'(KernelWidth - 1)));
  wire [0:0] valid_y_pos = (y_pos >= (YWidth'(KernelWidth - 1)));

  wire [0:0] valid_kernel_pos = valid_x_pos && valid_y_pos;

  wire [0:0] produce = valid_kernel_pos & in_fire;

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  // Provided Elastic State Machine Logic
  logic [0:0] valid_r;

  always_ff @(posedge clk_i) begin
    if (rst_i)        valid_r <= 1'b0;
    else if (ready_o) valid_r <= produce;
  end

  assign valid_o =  valid_r;
  assign ready_o = ~valid_r | ready_i;

  /* ------------------------------------ FIFO RAM Instantiations ------------------------------------ */
  // Both RAMs step forward with each new pixel input (in_fire) so that delay is exactly one row each.
  // RAM 1 produces a delay of one row. Reads off of data_i as the bottom row of the kernel receives new data.
  logic [WidthIn-1:0] ram_ol [KernelWidth];
  assign ram_ol[0] = data_i;

  generate
    for (genvar i = 0; i < KernelWidth - 1; i++) begin : gen_buffer
      delaybuffer #(
         .Width(WidthIn)
        ,.Delay (LineWidthPx - 1)
      ) ram_1_delaybuffer_inst (
         .clk_i  (clk_i)
        ,.rst_i  (rst_i)
        ,.data_i (ram_ol[i])
        ,.valid_i(in_fire)
        ,.ready_o()
        ,.valid_o()
        ,.data_o (ram_ol[i+1])
        ,.ready_i(1'b1)
      );
    end
  endgenerate
  /* ------------------------------------ Window Register Logic ------------------------------------ */
  // Row major Order(window[row][col])
  logic [WidthIn-1:0] window [KernelWidth][KernelWidth];

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
        for (int i = 0; i < KernelWidth; i++) begin
          for (int j = 0; j < KernelWidth; j++) begin
              window[i][j] <= '0;
          end
        end
    end else if (in_fire) begin
        /* ------------------------------- Internal Connections ------------------------------- */
        // Line feeds from right to left within window
        // Bottom line
        for (int r = 0; r < KernelWidth; r++) begin
          for (int c = 0; c < KernelWidth - 1; c++) begin
              window[r][c] <= window[r][c+1];
          end
        end
        /* -------------------------------- Input Connections -------------------------------- */
        // Load new data into the rightmost column of the window
        // Top line <- Twice seen data from second RAM
        for (int r = 0; r < KernelWidth; r++) begin
          window[r][KernelWidth-1] <= ram_ol[KernelWidth-1 - r];
        end
    end
  end
  /* ------------------------------------ Output Logic ------------------------------------ */
  logic signed [WidthOut-1:0] acc_l;
  always_comb begin
    acc_l = '0;
    for (int r = 0; r < KernelWidth; r++) begin
        for (int c = 0; c < KernelWidth; c++) begin
          // When binary inputs, only add the weight if the input pixel is a 1
          if (WidthIn == 1) begin
              if (window[r][c] != '0) begin
                acc_l = acc_l + $signed({{ExtendWidth{1'b0}}, weights_i[r*KernelWidth + c]});
              end
          end else begin
              acc_l = acc_l + ($signed(weights_i[r*KernelWidth + c]) * $signed(window[r][c]));
          end
        end
    end
  end
  assign data_o = acc_l;

endmodule
