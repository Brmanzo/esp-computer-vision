// conv_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module conv_layer #(
   parameter  int unsigned LineWidthPx    = 16
  ,parameter  int unsigned LineCountPx    = 12
  ,parameter  int unsigned InBits         = 1
  ,parameter  int unsigned OutBits        = 32
  ,parameter  int unsigned KernelWidth    = 3
  ,parameter  int unsigned WeightBits     = 2
  ,parameter  int unsigned InChannels     = 1
  ,parameter  int unsigned OutChannels    = 1
  ,localparam int unsigned KernelArea     = KernelWidth * KernelWidth

  ,localparam int unsigned TargetRamBits  = (LineWidthPx <= 255) ? 16 : 8
  ,localparam int unsigned ChannelsPerRam = TargetRamBits / ((KernelWidth - 1) * InBits)
  ,localparam int unsigned BufferCount    = (InChannels + ChannelsPerRam - 1) / ChannelsPerRam

  ,parameter  int unsigned Stride         = 1
  ,localparam int unsigned StrideBits     = (Stride <= 1) ? 1 : $clog2(Stride)

  ,localparam int XBits = (LineWidthPx <= 1) ? 1 : $clog2(LineWidthPx)
  ,localparam int YBits = (LineCountPx <= 1) ? 1 : $clog2(LineCountPx)

  ,localparam int unsigned WeightIndex = InChannels * KernelArea * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);
  /* ---------------------------------- Accumulator Width Calculation ----------------------- */
  // Generate only the minimum bits to support a full accumulation per filter
  function automatic int unsigned acc_bits;
    input int unsigned kernel_area, input_bits, weight_bits, in_channels;
    longint unsigned max_input, max_weight, worst_case_sum;
    begin
      max_input      = (64'd1 << input_bits) - 1;
      max_weight     = (64'd1 << (weight_bits - 1)) - 1;
      worst_case_sum = longint'(kernel_area) * max_input * max_weight * longint'(in_channels);

      acc_bits = $clog2(worst_case_sum + 1) + 1; // assign to function name
    end
  endfunction

  localparam MacBits = acc_bits(KernelArea, InBits, WeightBits, 1);
  localparam AccBits = acc_bits(KernelArea, InBits, WeightBits, InChannels);

  /* ---------------------------------------- Kernel Validation ---------------------------------------- */
  // Helper function to compute the next phase in the stride cycle for strides greater than 1.
  // Rolls over when the current phase is the last in the cycle, otherwise returns the next phase.
  function automatic logic [StrideBits-1:0] inc_stride(input logic [StrideBits-1:0] value);
    begin
      if (Stride <= 1) inc_stride = '0;
      else if (value == StrideBits'(Stride - 1)) inc_stride = '0;
      else inc_stride = value + StrideBits'(1);
    end
  endfunction
  
  wire [0:0] in_fire = valid_i & ready_o;

  // Position counters track the current x and y pixel positions within the input image.
  logic [XBits-1:0] x_pos;
  logic [YBits-1:0] y_pos;

  wire [0:0] valid_x_pos = (x_pos >= (XBits'(KernelWidth - 1)));
  wire [0:0] valid_y_pos = (y_pos >= (YBits'(KernelWidth - 1)));

  wire [0:0] last_col = (x_pos == XBits'(LineWidthPx - 1));
  wire [0:0] last_row = (y_pos == YBits'(LineCountPx - 1));

  // Stride phase counters track the current position within the stride cycle for x and y dimensions.
  logic [StrideBits-1:0] x_phase;
  logic [StrideBits-1:0] y_phase;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      x_pos <= '0;
      y_pos <= '0;
      x_phase <= '0;
      y_phase <= '0;
    // ---------------- Update Position Counters ----------------
    end else if (in_fire) begin
      if (last_col) begin
        x_pos <= '0;
        y_pos <= (last_row) ? '0 : (y_pos + 1);
      end else begin
        x_pos <= x_pos + 1;
      end
      // ---------------- Update Stride Phase Counters ----------------
      if (valid_x_pos) x_phase <= inc_stride(x_phase);
      // Reevaluate y stride phase each row
      if (last_col) begin
        // If end of row, reset x stride phase
        x_phase <= '0;
        if (valid_y_pos) y_phase <= inc_stride(y_phase);
        // If the end of the image, reset the y stride phase as well
        if (last_row) y_phase <= '0;
      end
    end
  end

  wire [0:0] valid_kernel_pos = valid_x_pos && valid_y_pos;

  wire [0:0] valid_x_stride = (Stride <= 1) ? 1'b1 : (x_phase == '0);
  wire [0:0] valid_y_stride = (Stride <= 1) ? 1'b1 : (y_phase == '0);
  wire [0:0] valid_stride = valid_x_stride && valid_y_stride;

  wire [0:0] produce = in_fire && valid_kernel_pos && valid_stride;

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  // Provided Elastic State Machine Logic
  logic [0:0] valid_r;

  always_ff @(posedge clk_i) begin
    if (rst_i)        valid_r <= 1'b0;
    else if (ready_o) valid_r <= produce;
  end

  assign valid_o =  valid_r;
  assign ready_o = ~valid_r | ready_i;

  /* --------------------------------------- Input Channel Logic --------------------------------------- */
  // Vertically partition channels and row buffers for each channel within RAM
  logic [InChannels-1:0][KernelWidth-1:0][InBits-1:0] row_buffers;
  logic [InChannels-1:0][KernelWidth-1:1][InBits-1:0] row_buffer_taps;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input
      assign row_buffers[ch][0] = data_i[ch]; // Row buffer 0 is current data input
      assign row_buffers[ch][KernelWidth-1:1] = row_buffer_taps[ch];
    end
  endgenerate

  /* ---------------------------------- Buffer Generation Logic----------------------- */
  // Targetting IceStorm's 30 4kB embedded block RAMs
  // https://www.mouser.com/datasheet/2/225/iCE40%20UltraPlus%20Family%20Data%20Sheet-1149905.pdf?srsltid=AfmBOoojsqUL7qv64GuzD_fsFp6UalE__EO5sBNN2KRE01qaez2zv7uA#page=15
  // 256 x 16, 512 x 8, 1024 x 4, or 2,048 x 2 bit configurations are possible with the 4kB RAMs

  // If buffer length exceeds 256, target 8 bit wide RAM
  //    max parameters for 1 channel:    3x3 kernel with 4 bit inputs or 9x9 kernel with 1 bit inputs
  //    max channels: 4 with parameters: 3x3 kernel with 1 bit inputs
  // If buffer length is 255 or less, target 16 bit wide RAM
  //    max parameters for 1 channel:    3x3 kernel with 8 bit inputs or 17x17 kernel with 1 bit inputs
  //    max channels: 8 with parameters: 3x3 kernel with 1 bit inputs

  logic [BufferCount-1:0][ChannelsPerRam-1:0][InBits-1:0] data_i_padded;
  logic [BufferCount-1:0][ChannelsPerRam-1:0][KernelWidth-1:1][InBits-1:0] data_o_padded;

  generate
    for (genvar buf_idx = 0; buf_idx < BufferCount; buf_idx++) begin : gen_ram_buffers
      // Generate the necessary number of buffers based on the input parameters and target RAM width
      localparam int unsigned FirstCh  = buf_idx * ChannelsPerRam;
      
      // Pad inputs so each RAM has a full set of channels
      for (genvar ch = 0; ch < ChannelsPerRam; ch++) begin : gen_padded_connections
        if (FirstCh + ch < InChannels) begin : gen_data_connections
          assign data_i_padded[buf_idx][ch] = data_i[FirstCh + ch];
          assign row_buffer_taps[FirstCh + ch] = data_o_padded[buf_idx][ch];
        end else begin : gen_zero_connections
          assign data_i_padded[buf_idx][ch] = '0;
        end
      end

      multi_delay_buffer #(
         .BufferWidth(InBits)
        ,.Delay      (LineWidthPx - 1)
        ,.BufferRows (KernelWidth - 1)
        ,.InputChannels(ChannelsPerRam)
      ) multi_delay_buffer_inst (
        .clk_i   (clk_i)
        ,.rst_i  (rst_i)

        ,.data_i (data_i_padded[buf_idx]) // Partition input channels across buffers
        ,.valid_i(in_fire)
        ,.ready_o()

        ,.data_o (data_o_padded[buf_idx]) // Row buffers >= 1 read from delay buffer
        ,.valid_o()
        ,.ready_i(1'b1)
      );
    end
  endgenerate
  /* verilator lint_on UNUSEDSIGNAL */

  /* ------------------------------------ Window Generation Logic ------------------------------------ */
  // Every input channel is represented within its own matrix and passed to every filter
  // Which each have input channel number of kernels 
  logic [InChannels-1:0][KernelArea-1:0][InBits-1:0] windows;

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
    end
  endgenerate

  /* ------------------------------------ Filter Logic ------------------------------------ */
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_row_buffer_delayed
      filter #(
        .InBits      (InBits)
        ,.OutBits    (OutBits)
        ,.KernelWidth(KernelWidth)
        ,.WeightBits (WeightBits)
        ,.MacBits    (MacBits)
        ,.AccBits    (AccBits)
        ,.InChannels (InChannels)
      ) filter_inst (
         .windows_i  (windows)
        ,.weights_i  (Weights[ch*WeightIndex +: WeightIndex])
        ,.data_o     (data_o[ch])
      );
    end
  endgenerate

endmodule
