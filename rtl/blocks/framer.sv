`timescale 1ns / 1ps

module framer #(
   parameter int unsigned  UnpackedWidth  = 1
  ,parameter int unsigned  PackedNum      = 8
  ,parameter int unsigned  PackedWidth    = UnpackedWidth * PackedNum
  ,parameter int unsigned  PacketLenElems = 1024 // Number of packed elements per packet

  ,parameter  logic [PackedWidth-1:0] TailByte0 = PackedWidth'($unsigned(165)) // 0xA5
  ,parameter  logic [PackedWidth-1:0] TailByte1 = PackedWidth'($unsigned(90))  // 0x5A

  ,localparam int unsigned DimensionWidth = PackedWidth * 2 // Width to hold image dimensions
  ,localparam int unsigned CountWidth     = $clog2(PacketLenElems)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0]                valid_i
  ,output [0:0]                ready_o
  ,input  [UnpackedWidth-1:0]  unpacked_i
  ,input  [DimensionWidth-1:0] image_height_i
  ,input  [DimensionWidth-1:0] image_width_i

  ,output [0:0]             valid_o
  ,input  [0:0]             ready_i
  ,output [PackedWidth-1:0] data_o
);
  // FSM States
  typedef enum logic [2:0] {Forward, Footer0, Footer1, ImageWH, ImageWL, ImageHH, ImageHL} fsm_e;
  fsm_e state_q, state_d;

  // Packer Wires
  wire  [0:0]             pack_ready;
  wire  [0:0]             pack_valid;
  wire  [PackedWidth-1:0] packed_data;

  // Handshake Wires
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  /* ---------------------------------------- Counter Logic ---------------------------------------- */
  logic [CountWidth-1:0]               counter_q, counter_d;
  wire  [CountWidth-1:0] max_count   = CountWidth'(PacketLenElems - 1);
  wire  [0:0]            counter_max = (counter_q == max_count);

  // Saturating counter to track number of packed inputs
  always_ff @(posedge clk_i) begin
    if (rst_i) counter_q <= '0;
    else       counter_q <= counter_d;
  end

  always_comb begin
    counter_d = counter_q; // Default hold
    // Reset counter when finishing footer
    if (state_q == ImageHL && out_fire) begin
      counter_d = '0;
    // Increment counter when accepting input in forward state
    end else if (state_q == Forward && in_fire) begin
      // Saturate at max count
      if (!counter_max) counter_d = counter_q + 1'b1;
    end
  end

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  wire  [0:0] last_input = in_fire && counter_max;
  logic [0:0] flushing;

  logic [0:0] rx_complete;
  wire  [0:0] tx_complete = rx_complete && out_fire;

    // Flushing logic to flush packer when receiving last input
  always_ff @(posedge clk_i) begin
    if (rst_i) flushing <= 1'b0;
    else       flushing <= last_input;
  end

  // RX Complete logic to track when a full packet has been received
  always_ff @(posedge clk_i) begin
    if (rst_i)                               rx_complete <= 1'b0;
    else if (state_q == ImageHL && out_fire) rx_complete <= 1'b0;
    else if (last_input)                     rx_complete <= 1'b1;
  end

  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q <= Forward;
      data_q  <= '0;
      valid_q <= 1'b0;
    end else begin
      data_q  <= data_d;
      valid_q <= valid_d;
      if (out_fire) begin
        state_q <= state_d;
      end
    end
  end

  // Next state logic
  always_comb begin
    state_d = state_q;
    case (state_q)
      Forward: if (tx_complete) state_d = Footer0;
      Footer0: if (out_fire)    state_d = Footer1;
      Footer1: if (out_fire)    state_d = ImageWH;
      ImageWH: if (out_fire)    state_d = ImageWL;
      ImageWL: if (out_fire)    state_d = ImageHH;
      ImageHH: if (out_fire)    state_d = ImageHL;
      ImageHL: if (out_fire)    state_d = Forward;
      default:                  state_d = Forward;
    endcase
  end

  // Data output is either packed data or footer bytes
  logic [PackedWidth-1:0] data_q,  data_d;
  assign data_o  = data_d;
  // Valid output unless packer is not valid in forward state
  logic [0:0]             valid_q, valid_d;
  assign valid_o = valid_d;
  // Forward data from packer until footer state, then send tail bytes
  // Only ready to accept new input data in forward state and not complete
  assign ready_o = (state_q == Forward) && pack_ready && !rx_complete;

  // Data Logic
  always_comb begin
    data_d  = data_q;  // Default hold
    valid_d = valid_q; // Default hold
    case (state_q)
      Forward: begin data_d = packed_data;                                  valid_d = pack_valid; end
      Footer0: begin data_d = TailByte0;                                    valid_d = 1'b1;       end
      Footer1: begin data_d = TailByte1;                                    valid_d = 1'b1;       end
      ImageWH: begin data_d = image_width_i [DimensionWidth-1:PackedWidth]; valid_d = 1'b1;       end
      ImageWL: begin data_d = image_width_i [PackedWidth-1:0];              valid_d = 1'b1;       end
      ImageHH: begin data_d = image_height_i[DimensionWidth-1:PackedWidth]; valid_d = 1'b1;       end
      ImageHL: begin data_d = image_height_i[PackedWidth-1:0];              valid_d = 1'b1;       end
      default: begin data_d = data_q;                                       valid_d = valid_q;    end
    endcase
  end

  /* ------------------------------------------ Packer Inst ------------------------------------------ */
  // Packer to pack 4 2-bit magnitude values into each 8-bit UART output
  packer #(
     .UnpackedWidth(UnpackedWidth)
    ,.PackedNum    (PackedNum)
  ) packer_inst (
     .clk_i(clk_i)
    ,.rst_i(rst_i)

    ,.unpacked_i(unpacked_i)
    ,.flush_i   (flushing)
    ,.valid_i   (valid_i && !rx_complete)
    ,.ready_o   (pack_ready)

    ,.packed_o  (packed_data)
    ,.valid_o   (pack_valid)
    ,.ready_i   ((state_q == Forward) ? ready_i : 1'b0) // Only accept output in forward state
  );

endmodule
