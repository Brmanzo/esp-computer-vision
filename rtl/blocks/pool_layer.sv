// pool_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module pool_layer #(
    parameter int unsigned LineWidthPx = 16,
    parameter int unsigned LineCountPx = 12,
    parameter int unsigned InBits      = 1,
    parameter int unsigned OutBits     = 1,
    parameter int unsigned KernelWidth = 2,
    parameter int unsigned InChannels  = 1,
    parameter int unsigned PoolMode    = 0
) (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [InChannels-1:0][OutBits-1:0] data_o
);

  localparam int unsigned OutChannels = InChannels;
  localparam int unsigned KernelArea  = KernelWidth * KernelWidth;

  localparam int unsigned TargetRamBits  = (LineWidthPx <= 255) ? 16 : 8;
  localparam int unsigned KernelSize     = (((KernelWidth - 1) * InBits) > 0) ? ((KernelWidth - 1) * InBits) : 1;
  localparam int unsigned ChannelsPerRam = ((TargetRamBits / KernelSize) > 0) ? (TargetRamBits / KernelSize) : 1;
  localparam int unsigned BufferCount    = (InChannels + ChannelsPerRam - 1) / ChannelsPerRam;

  localparam int unsigned Stride     = KernelWidth;
  localparam int unsigned StrideBits = (Stride <= 1) ? 1 : $clog2(Stride);

  localparam int XBits = (LineWidthPx <= 1) ? 1 : $clog2(LineWidthPx);
  localparam int YBits = (LineCountPx <= 1) ? 1 : $clog2(LineCountPx);

  // Helper function to compute the next phase in the stride cycle for strides greater than 1.
  function automatic logic [StrideBits-1:0] inc_stride(input logic [StrideBits-1:0] value);
    begin
      if (Stride <= 1) inc_stride = '0;
      else if (value == StrideBits'(Stride - 1)) inc_stride = '0;
      else inc_stride = value + StrideBits'(1);
    end
  endfunction

  /* ---------------------------------------- Kernel Validation ---------------------------------------- */
  wire [0:0] in_fire = valid_i & ready_o;

  logic [XBits-1:0] x_pos;
  logic [YBits-1:0] y_pos;

  wire [0:0] valid_x_pos = (x_pos >= (XBits'(KernelWidth - 1)));
  wire [0:0] valid_y_pos = (y_pos >= (YBits'(KernelWidth - 1)));

  wire [0:0] last_col = (x_pos == XBits'(LineWidthPx - 1));
  wire [0:0] last_row = (y_pos == YBits'(LineCountPx - 1));

  logic [StrideBits-1:0] x_phase;
  logic [StrideBits-1:0] y_phase;

  wire [0:0] valid_x_stride = (Stride <= 1) ? 1'b1 : (x_phase == '0);
  wire [0:0] valid_y_stride = (Stride <= 1) ? 1'b1 : (y_phase == '0);

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      x_pos <= '0;
      y_pos <= '0;
      x_phase <= '0;
      y_phase <= '0;
    end else if (in_fire) begin
      if (last_col) begin
        x_pos <= '0;
        y_pos <= (last_row) ? '0 : (y_pos + 1);
      end else begin
        x_pos <= x_pos + 1;
      end
      if (valid_x_pos) x_phase <= inc_stride(x_phase);
      if (last_col) begin
        x_phase <= '0;
        if (valid_y_pos) y_phase <= inc_stride(y_phase);
        if (last_row) y_phase <= '0;
      end
    end
  end

  wire [0:0] valid_kernel_pos = valid_x_pos && valid_y_pos;
  wire [0:0] produce = valid_kernel_pos & in_fire && valid_x_stride && valid_y_stride;

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  logic [0:0] valid_r;
  always_ff @(posedge clk_i) begin
    if (rst_i)        valid_r <= 1'b0;
    else if (ready_o) valid_r <= produce;
  end

  assign valid_o =  valid_r;
  assign ready_o = ~valid_r | ready_i;

  /* --------------------------------------- Input Channel Logic --------------------------------------- */
  logic signed [InChannels-1:0][KernelWidth-1:0][InBits-1:0] row_buffers;
  logic signed [InChannels-1:0][KernelWidth-1:1][InBits-1:0] row_buffer_taps;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input
      assign row_buffers[ch][0] = data_i[ch];
      assign row_buffers[ch][KernelWidth-1:1] = row_buffer_taps[ch];
    end
  endgenerate

  multi_delay_ram #(
     .BufferCount    (BufferCount)
    ,.ChannelsPerRam (ChannelsPerRam)
    ,.InBits         (InBits)
    ,.InChannels     (InChannels)
    ,.KernelWidth    (KernelWidth)
    ,.LineWidthPx    (LineWidthPx)
  ) multi_delay_ram_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.in_fire (in_fire)
    ,.data_i  (data_i)
    ,.data_o  (row_buffer_taps)
  );

  /* ------------------------------------ Window Generation Logic ------------------------------------ */
  logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0] windows;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_windows
      window #(
         .KernelWidth(KernelWidth)
        ,.InBits    (InBits)
      ) win_i (
         .clk_i   (clk_i)
        ,.rst_i   (rst_i)
        ,.in_fire_i    (in_fire)
        ,.row_buffers_i(row_buffers[ch])
        ,.window_o     (windows[ch])
      );
      if (PoolMode == 0) begin : gen_max_pool
        max #(
           .KernelWidth(KernelWidth)
          ,.InBits   (InBits)
        ) max_i (
           .window(windows[ch])
          ,.data_o(data_o[ch])
        );
      end else begin : gen_avg_pool
        avg #(
           .KernelWidth(KernelWidth)
          ,.InBits    (InBits)
        ) avg_i (
           .window(windows[ch])
          ,.data_o(data_o[ch])
        );
      end
    end
  endgenerate

endmodule
