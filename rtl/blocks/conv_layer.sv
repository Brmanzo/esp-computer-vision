// conv_layer.sv
// Bradley Manzo, 2026
  
/* verilator lint_off PINCONNECTEMPTY */
`timescale 1ns / 1ps
module conv_layer #(
   parameter  int unsigned LineWidthPx    = 16
  ,parameter  int unsigned LineCountPx    = 12
  ,parameter  int unsigned InBits         = 1
  ,parameter  int unsigned OutBits        = 1
  ,parameter  int unsigned KernelWidth    = 3
  ,parameter  int unsigned WeightBits     = 2
  ,parameter  int unsigned InChannels     = 1
  ,parameter  int unsigned OutChannels    = 1
  ,localparam int unsigned KernelArea     = KernelWidth * KernelWidth

  ,localparam int unsigned TargetRamBits  = ((LineWidthPx - 1) <= 256) ? 16 : 8
  ,localparam int unsigned ChannelsPerRam = TargetRamBits / ((KernelWidth - 1) * InBits)
  ,localparam int unsigned BufferCount    = (InChannels + ChannelsPerRam - 1) / ChannelsPerRam

  ,parameter  int unsigned Stride         = 1 // TODO Add Padding support
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
  ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

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
      max_input      = (64'd1 << (input_bits - 1));
      max_weight     = (64'd1 << (weight_bits - 1)) - 1;
      worst_case_sum = longint'(kernel_area) * max_input * max_weight * longint'(in_channels);

      acc_bits = $clog2(worst_case_sum + 1) + 1; // assign to function name
    end
  endfunction

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
  logic  [0:0] input_valid_r;

  always_ff @(posedge clk_i) begin
    // If reset, we are not valid this cycle
    if (rst_i)        input_valid_r <= 1'b0; 
    // If not stalling, then we have valid data if we are producing it this cycle
    else if (ready_o) input_valid_r <= produce;
  end

  wire [0:0] window_valid;

  // Each filter has its own internal pipeline stage
  wire [OutChannels-1:0] filter_ready;
  wire [OutChannels-1:0] filter_valid;
  // Top level output is valid and ready when all filters are valid and ready, respectively
  assign valid_o = &filter_valid;
  
  /* --------------------------------------- Input Channel Logic --------------------------------------- */
  // Vertically partition channels and row buffers for each channel within RAM
  logic signed [InChannels-1:0][KernelWidth-1:0][InBits-1:0] row_buffers;
  logic signed [InChannels-1:0][KernelWidth-1:1][InBits-1:0] row_buffer_taps;
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_data_input
      assign row_buffers[ch][0] = $signed(data_i[ch]); // Row buffer 0 is current data input
      assign row_buffers[ch][KernelWidth-1:1] = row_buffer_taps[ch];
    end
  endgenerate

  /* ---------------------------------- Buffer Generation Logic----------------------- */
  // Maps the input channel delay buffers to IceStorm's 30 4kB block RAMs

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
    ,.data_o  (row_buffer_taps) // Row buffers >= 1 read from delay buffer
  );

  /* ------------------------------------ Window Generation Logic ------------------------------------ */
  // Every input channel is represented within its own matrix and passed to every filter
  // Which each have input channel number of kernels 
  logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0] windows_d, windows_q;

  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_windows
      window #(
         .KernelWidth(KernelWidth)
        ,.InBits    (InBits)
      ) window_inst (
         .clk_i   (clk_i)
        ,.rst_i   (rst_i)

        ,.in_fire_i    (in_fire)
        ,.row_buffers_i(row_buffers[ch])
        ,.window_o     (windows_d[ch])
      );
    end
  endgenerate

  // Delay between Window output and filter input
  elastic #(
     .InBits        (InChannels * KernelArea * InBits)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)

    ,.valid_i (input_valid_r) // Valid when top-level is valid
    ,.ready_o (ready_o)       // Handles top-level backpressure
    ,.data_i  (windows_d)

    ,.valid_o (window_valid)
    ,.ready_i (&filter_ready) // ready when all filters are ready to accept data
    ,.data_o  (windows_q)
  );

  /* ------------------------------------ Filter Logic ------------------------------------ */
  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_row_buffer_delayed
      filter #(
         .InBits      (InBits)
        ,.OutBits    (OutBits)
        ,.KernelWidth(KernelWidth)
        ,.WeightBits (WeightBits)
        ,.AccBits    (AccBits)
        ,.InChannels (InChannels)
      ) filter_inst (
         .clk_i      (clk_i)
        ,.rst_i      (rst_i)
        ,.valid_i    (window_valid)
        ,.ready_o    (filter_ready[ch])
        ,.windows_i  (windows_q)
        ,.weights_i  (Weights[ch*WeightIndex +: WeightIndex])
        ,.ready_i    (ready_i)
        ,.valid_o    (filter_valid[ch])
        ,.data_o     (data_o[ch])
      );
    end
  endgenerate

endmodule
