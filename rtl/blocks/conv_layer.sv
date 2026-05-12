// conv_layer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps

/* ---------------------------------- Accumulator Width Calculation ----------------------- */
function automatic int unsigned conv_acc_bits(
    input int unsigned kernel_area, 
    input int unsigned input_bits, 
    input int unsigned weight_bits, 
    input int unsigned in_channels, 
    input int unsigned bias_bits
);
  longint unsigned max_input, max_weight, worst_case_sum;
  int unsigned wc_bits;
  begin
    max_input = (input_bits <= 2) ? 64'd1 : (64'd1 << (input_bits - 1));
    max_weight = (weight_bits <= 2) ? 64'd1 : (64'd1 << (weight_bits - 1));
    worst_case_sum = longint'(kernel_area) * max_input * max_weight * longint'(in_channels);
    wc_bits = $clog2(worst_case_sum + 1) + 1;
    wc_bits = ((wc_bits > bias_bits) ? wc_bits : bias_bits) + 1;
    conv_acc_bits = (wc_bits > 32) ? 32 : wc_bits;
  end
endfunction

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

   ,parameter  int unsigned Stride     = 1
   ,localparam int unsigned StrideBits = (Stride <= 1) ? 1 : $clog2(Stride)

   ,parameter int unsigned Padding  = 0
   ,parameter int unsigned DSPCount = 0 // 0: LUT, n>=1: Sequential DSP Allocation

   ,localparam int unsigned PaddedWidth  = LineWidthPx + (2 * Padding)
   ,localparam int unsigned PaddedHeight = LineCountPx + (2 * Padding)
   ,localparam int XBits = (LineWidthPx <= 1) ? 1 : $clog2(PaddedWidth + 1)
   ,localparam int YBits = (LineCountPx <= 1) ? 1 : $clog2(PaddedHeight + 1)

   ,localparam int unsigned WeightIndex = InChannels * KernelArea * WeightBits
   ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
   ,parameter logic signed [OutChannels*BiasBits-1:0] Biases = '0
`ifdef VERILATOR
   ,parameter string FileName = ""
   ,parameter string FileName_0 = ""
`else
   ,parameter [8*256-1:0] FileName = ""
   ,parameter [8*256-1:0] FileName_0 = ""
`endif
) (
    input  [0:0] clk_i
   ,input  [0:0] rst_i

   ,input  [0:0] valid_i
   ,output [0:0] ready_o
   ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

   ,output [0:0] valid_o
   ,input  [0:0] ready_i
   ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

  localparam AccBits = conv_acc_bits(KernelArea, InBits, WeightBits, InChannels, BiasBits);

  /* ---------------------------------------- Kernel Validation ---------------------------------------- */
  logic [XBits-1:0] x_pos;
  logic [YBits-1:0] y_pos;

  wire  [0:0] valid_x_pos = (x_pos >= (XBits'(KernelWidth - 1)));
  wire  [0:0] valid_y_pos = (y_pos >= (YBits'(KernelWidth - 1)));
  wire  [0:0] valid_kernel_pos = valid_x_pos && valid_y_pos;

  wire  [0:0] last_col    = (x_pos == XBits'((LineWidthPx - 1) + 2 * Padding));
  wire  [0:0] last_row    = (y_pos == YBits'((LineCountPx - 1) + 2 * Padding));

  /* ------------------------------------------ Stride Logic ------------------------------------------ */
  logic [StrideBits-1:0] x_phase;
  logic [StrideBits-1:0] y_phase;
  wire  [0:0] valid_stride = ((Stride <= 1) ? 1'b1 : (x_phase == '0)) && ((Stride <= 1) ? 1'b1 : (y_phase == '0));

  /* ----------------------------------------- Padding Logic ----------------------------------------- */
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
      x_pos <= '0; y_pos <= '0; x_phase <= '0; y_phase <= '0;
    end else if (in_fire) begin
      // If last column, reset to next row of first column
      if (last_col) begin
        x_pos <= '0;
        y_pos <= (last_row) ? '0 : (y_pos + 1);
        x_phase <= '0;
        // If we're striding this row, emit a zero
        if (valid_y_pos) begin
           if (Stride > 1 && y_phase == StrideBits'(Stride - 1)) y_phase <= '0;
           else if (Stride > 1) y_phase <= y_phase + StrideBits'(1);
        end
        // If last pixel, also reset to first row
        if (last_row) y_phase <= '0;
      // Otherwise, increment column by 1
      end else begin
        x_pos <= x_pos + 1;
        // If we're striding, emit a zero
        if (valid_x_pos) begin
           if (Stride > 1 && x_phase == StrideBits'(Stride - 1)) x_phase <= '0;
           else if (Stride > 1) x_phase <= x_phase + StrideBits'(1);
        end
      end
    end
  end

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  logic [0:0] valid_q;
  wire  [0:0] produce = in_fire && valid_kernel_pos && valid_stride;
  
  always_ff @(posedge clk_i) begin
    if (rst_i) valid_q <= 1'b0;
    else if (elastic_ready) valid_q <= produce;
  end

  logic [InChannels-1:0][KernelWidth-1:0][InBits-1:0] row_buffers;
  logic signed [InChannels-1:0][InBits-1:0] padded_data_i;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input
      // If padding this current cycle, insert a zero to buffer, otherwise connect to data_i
      assign padded_data_i[ch] = pad_cycle ? '0 : $signed(data_i[ch]);
      // Duplicate onto head of row buffers
      assign row_buffers[ch][0] = padded_data_i[ch];
    end

    // If kernel is greater than 1x1, then we buffer the previous rows
    // For every Input Channel, for every row buffer, vertically partitioned
    // In a RAM entry. Yosys maps cleanly to Icestorm's 4KB BRAM.
    if (KernelWidth > 1) begin : gen_delay_ram
      localparam int unsigned ChannelDelayBits = (KernelWidth - 1) * InBits;
      logic [InChannels * ChannelDelayBits - 1 : 0] row_buffer_taps;
      for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input_taps
        for (genvar k = 1; k < KernelWidth; k++) begin : gen_row_taps
          assign row_buffers[ch][k] = row_buffer_taps[(ch*ChannelDelayBits + (k-1)*InBits) +: InBits];
        end
      end
      /* verilator lint_off PINCONNECTEMPTY */
      multi_delay_buffer #(
         .BufferWidth  (InBits)
        ,.Delay        (PaddedWidth - 1)
        ,.BufferRows   (KernelWidth - 1)
        ,.InputChannels(InChannels)
      ) multi_delay_buffer_inst (
         .clk_i   (clk_i)
        ,.rst_i   (rst_i)

        ,.data_i  (padded_data_i)
        ,.valid_i (in_fire)
        ,.ready_o ()

        ,.data_o  (row_buffer_taps)
        ,.valid_o ()
        ,.ready_i (1'b1)
      );
    end
  endgenerate

  /* ------------------------------------ Window Generation Logic ------------------------------------ */
  wire [0:0] window_valid;
  logic signed [InChannels*KernelArea*InBits-1:0] windows_d, windows_q;
  wire [0:0] filter_ready_combined;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_windows
      window #(
         .KernelWidth(KernelWidth)
        ,.InBits     (InBits)
      ) window_inst (
         .clk_i         (clk_i)
        ,.rst_i         (rst_i)
        ,.in_fire_i     (in_fire)
        ,.row_buffers_i (row_buffers[ch])
        ,.window_o      (windows_d[ch*KernelArea*InBits +: KernelArea*InBits])
      );
    end
  endgenerate

  assign window_valid = valid_q;
  assign elastic_ready = filter_ready_combined;
  assign windows_q = windows_d;

  /* ------------------------------------ Filter Logic ------------------------------------ */
  generate
    if (DSPCount > 0) begin : gen_seq_filters
      filter_seq #(
         .DSPCount    (DSPCount)
        ,.InBits      (InBits)
        ,.OutBits     (OutBits)
        ,.KernelWidth (KernelWidth)
        ,.WeightBits  (WeightBits)
        ,.BiasBits    (BiasBits)
        ,.AccBits     (AccBits)
        ,.InChannels  (InChannels)
        ,.OutChannels (OutChannels)
        ,.Biases      (Biases)
        ,.FileName    (FileName)
      ) filter_seq_inst (
         .clk_i    (clk_i)
        ,.rst_i    (rst_i)
        ,.valid_i  (window_valid)
        ,.ready_o  (filter_ready_combined)
        ,.windows_i(windows_q)
        ,.ready_i  (ready_i)
        ,.valid_o  (valid_o)
        ,.data_o   (data_o)
      );
    end else begin : gen_parallel_filters
      wire [OutChannels-1:0] f_ready, f_valid;
      for (genvar oc = 0; oc < OutChannels; oc++) begin : gen_each_filter
        filter #(
             .InBits      (InBits)
            ,.OutBits     (OutBits)
            ,.KernelWidth (KernelWidth)
            ,.WeightBits  (WeightBits)
            ,.AccBits     (AccBits)
            ,.InChannels  (InChannels)
            ,.Bias        (Biases[oc*BiasBits+:BiasBits])
        ) filter_inst (
             .clk_i     (clk_i)
            ,.rst_i     (rst_i)
            ,.valid_i   (window_valid)
            ,.ready_o   (f_ready[oc])
            ,.windows_i (windows_q)
            ,.weights_i (Weights[oc*WeightIndex+:WeightIndex])
            ,.ready_i   (ready_i)
            ,.valid_o   (f_valid[oc])
            ,.data_o    (data_o[oc])
        );
      end
      assign valid_o = &f_valid;
      assign filter_ready_combined = &f_ready;
    end
  endgenerate

endmodule
