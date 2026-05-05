// conv_layer.sv
// Bradley Manzo, 2026

/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module conv_layer #(
   parameter  int unsigned LineWidthPx = 16
  ,parameter  int unsigned LineCountPx = 12
  ,parameter  int unsigned InBits      = 1
  ,parameter  int unsigned OutBits     = 1
  ,parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WeightBits  = 2
  ,parameter  int unsigned BiasBits    = 8
  ,parameter  int unsigned InChannels  = 1
  ,parameter  int unsigned OutChannels = 1
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth

  ,localparam int unsigned TargetRamBits = ((LineWidthPx - 1) <= 256) ? 16 : 8
  ,localparam int unsigned KernelSize     = (((KernelWidth - 1) * InBits) > 0) ? ((KernelWidth - 1) * InBits) : 1
  ,localparam int unsigned ChannelsPerRam = ((TargetRamBits / KernelSize) > 0) ? (TargetRamBits / KernelSize) : 1
  ,localparam int unsigned BufferCount = (InChannels + ChannelsPerRam - 1) / ChannelsPerRam

  ,parameter  int unsigned Stride     = 1
  ,localparam int unsigned StrideBits = (Stride <= 1) ? 1 : $clog2(Stride)

  ,parameter int unsigned Padding = 0
  ,parameter int unsigned UseDSP  = 0 // Set to 1 to use sequential DSP filters

  ,localparam int unsigned PaddedWidth  = LineWidthPx + (2 * Padding)
  ,localparam int unsigned PaddedHeight = LineCountPx + (2 * Padding)
  ,localparam int XBits = (LineWidthPx <= 1) ? 1 : $clog2(PaddedWidth + 1)
  ,localparam int YBits = (LineCountPx <= 1) ? 1 : $clog2(PaddedHeight + 1)

  ,localparam int unsigned WeightIndex = InChannels * KernelArea * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0

  ,parameter logic signed [OutChannels*BiasBits-1:0] Biases = '0
) (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0] valid_i
  ,output [0:0] ready_o
  ,input logic signed [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i

  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);
  /* ---------------------------------- Accumulator Width Calculation ----------------------- */
  // Generate only the minimum bits to support a full accumulation per filter
  function automatic int unsigned acc_bits;
    input int unsigned kernel_area, input_bits, weight_bits, in_channels, bias_bits;
    longint unsigned max_input, max_weight, worst_case_sum;
    int unsigned wc_bits;
    begin
      max_input = (input_bits <= 2) ? 64'd1 : (64'd1 << (input_bits - 1));
      max_weight = (weight_bits <= 2) ? 64'd1 : (64'd1 << (weight_bits - 1));
      worst_case_sum = longint'(kernel_area) * max_input * max_weight * longint'(in_channels);

      wc_bits = $clog2(worst_case_sum + 1) + 1;  // assign to function name
      acc_bits = ((wc_bits > bias_bits) ? wc_bits : bias_bits) + 1; // Ensure we can also represent the bias
    end
  endfunction

  localparam AccBits = acc_bits(KernelArea, InBits, WeightBits, InChannels, BiasBits);

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

  // Position counters track the current x and y pixel positions within the input image.
  logic [XBits-1:0] x_pos;
  logic [YBits-1:0] y_pos;

  // Valid positions of kernel within input activation, (excludes invalid buffering states)
  wire [0:0] valid_x_pos = (x_pos >= (XBits'(KernelWidth - 1)));
  wire [0:0] valid_y_pos = (y_pos >= (YBits'(KernelWidth - 1)));

  wire [0:0] valid_kernel_pos = valid_x_pos && valid_y_pos;

  // Last column and row indicators for resetting position counters
  wire [0:0] last_col = (x_pos == XBits'((LineWidthPx - 1) + 2 * Padding));
  wire [0:0] last_row = (y_pos == YBits'((LineCountPx - 1) + 2 * Padding));

  /* ------------------------------------------ Stride Logic ------------------------------------------ */
  // Stride phase counters track the current position within the stride cycle for x and y dimensions.
  logic [StrideBits-1:0] x_phase;
  logic [StrideBits-1:0] y_phase;

  wire [0:0] valid_x_stride = (Stride <= 1) ? 1'b1 : (x_phase == '0);
  wire [0:0] valid_y_stride = (Stride <= 1) ? 1'b1 : (y_phase == '0);

  wire [0:0] valid_stride = valid_x_stride && valid_y_stride;

  /* ----------------------------------------- Padding Logic ----------------------------------------- */
  // Padding (If we are before the first real pixel, or after the last real pixel)
  /* verilator lint_off UNSIGNED */
  wire [0:0] pad_x = (x_pos < XBits'(Padding)) | (x_pos >= XBits'(LineWidthPx + Padding));
  wire [0:0] pad_y = (y_pos < YBits'(Padding)) | (y_pos >= YBits'(LineCountPx + Padding));
  /* verilator lint_on UNSIGNED */
  wire [0:0] pad_cycle = pad_x | pad_y;

  wire [0:0] elastic_ready;
  wire [0:0] in_fire = (valid_i | pad_cycle) & elastic_ready;
  assign ready_o = elastic_ready & ~pad_cycle;

  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      x_pos   <= '0;
      y_pos   <= '0;
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

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  // Provided Elastic State Machine Logic
  logic [0:0] input_valid_r;
  wire  [0:0] produce = in_fire && valid_kernel_pos && valid_stride;

  always_ff @(posedge clk_i) begin
    // If reset, we are not valid this cycle
    if (rst_i) input_valid_r <= 1'b0;
    // If not stalling, then we have valid data if we are producing it this cycle
    else if (elastic_ready) input_valid_r <= produce;
  end

  // Each filter has its own internal pipeline stage
  wire [OutChannels-1:0] filter_ready;
  wire [OutChannels-1:0] filter_valid;
  // Top level output is valid when all filters are valid
  assign valid_o = &filter_valid;

  /* --------------------------------------- Input Channel Logic --------------------------------------- */
  // Vertically partition channels and row buffers for each channel within RAM
  logic signed [InChannels-1:0][KernelWidth-1:0][InBits-1:0] row_buffers;
  logic signed [InChannels-1:0][InBits-1:0] padded_data_i;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input
      assign padded_data_i[ch] = pad_cycle ? '0 : $signed(
              data_i[ch]
          );  // If padding, input 0, else input data
      assign row_buffers[ch][0] = padded_data_i[ch];  // Row buffer 0 is current data input
    end

    // Only buffer if kernel exceeds 1x1, (if so row_buffers is simply data_i)
    if (KernelWidth > 1) begin : gen_delay_ram
      localparam int unsigned ChannelDelayBits = (KernelWidth - 1) * InBits;
      logic [InChannels * ChannelDelayBits - 1 : 0] row_buffer_taps;

      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input_taps
        for (genvar k = 1; k < KernelWidth; k++) begin : gen_row_taps
          assign row_buffers[ch][k] = row_buffer_taps[(ch*ChannelDelayBits + (k-1)*InBits) +: InBits];
        end
      end

      /* ---------------------------------- Buffer Generation Logic----------------------- */
      // Maps the input channel delay buffers to IceStorm's 30 4kB block RAMs
      multi_delay_ram #(
            .BufferCount   (BufferCount)
          , .ChannelsPerRam(ChannelsPerRam)
          , .InBits        (InBits)
          , .InChannels    (InChannels)
          , .KernelWidth   (KernelWidth)
          , .LineWidthPx   (PaddedWidth)  // Account for padding in line width
      ) multi_delay_ram_inst (
            .clk_i  (clk_i)
          , .rst_i  (rst_i)
          , .in_fire(in_fire)
          , .data_i (padded_data_i)
          , .data_o (row_buffer_taps)  // Row buffers >= 1 read from delay buffer
      );
    end
  endgenerate

  /* ------------------------------------ Window Generation Logic ------------------------------------ */
  // Every input channel is represented within its own matrix and passed to every filter
  // Which each have input channel number of kernels 
  wire [0:0] window_valid;
  logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0] windows_d, windows_q;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_windows
      window #(
         .KernelWidth(KernelWidth)
        ,.InBits     (InBits)
      ) window_inst (
         .clk_i(clk_i)
        ,.rst_i(rst_i)

        ,.in_fire_i    (in_fire)
        ,.row_buffers_i(row_buffers[ch])
        ,.window_o     (windows_d[ch])
      );
    end
  endgenerate

  // Delay between Window output and filter input
  elastic #(
     .InBits       (InChannels * KernelArea * InBits)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i(clk_i)
    ,.rst_i(rst_i)

    ,.valid_i(input_valid_r)  // Valid when top-level is valid
    ,.ready_o(elastic_ready)  // Handles top-level backpressure
    ,.data_i (windows_d)

    ,.valid_o(window_valid)
    ,.ready_i(&filter_ready)  // ready when all filters are ready to accept data
    ,.data_o (windows_q)
  );
  /* ------------------------------------ Filter Logic ------------------------------------ */
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_row_buffer_delayed
      if (UseDSP) begin : gen_filter_dsp
        filter_dsp #(
           .InBits     (InBits)
          ,.OutBits    (OutBits)
          ,.KernelWidth(KernelWidth)
          ,.WeightBits (WeightBits)
          ,.AccBits    (AccBits)
          ,.InChannels (InChannels)
          ,.Bias       (Biases[ch*BiasBits+:BiasBits])
        ) filter_inst (
           .clk_i    (clk_i)
          ,.rst_i    (rst_i)
          ,.valid_i  (window_valid)
          ,.ready_o  (filter_ready[ch])
          ,.windows_i(windows_q)
          ,.weights_i(Weights[ch*WeightIndex+:WeightIndex])
          ,.ready_i  (ready_i)
          ,.valid_o  (filter_valid[ch])
          ,.data_o   (data_o[ch])
        );
      end else begin : gen_filter_lut
        filter #(
           .InBits     (InBits)
          ,.OutBits    (OutBits)
          ,.KernelWidth(KernelWidth)
          ,.WeightBits (WeightBits)
          ,.AccBits    (AccBits)
          ,.InChannels (InChannels)
          ,.Bias       (Biases[ch*BiasBits+:BiasBits])
        ) filter_inst (
           .clk_i    (clk_i)
          ,.rst_i    (rst_i)
          ,.valid_i  (window_valid)
          ,.ready_o  (filter_ready[ch])
          ,.windows_i(windows_q)
          ,.weights_i(Weights[ch*WeightIndex+:WeightIndex])
          ,.ready_i  (ready_i)
          ,.valid_o  (filter_valid[ch])
          ,.data_o   (data_o[ch])
        );
      end
    end
  endgenerate

endmodule
