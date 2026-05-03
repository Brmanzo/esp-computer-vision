// framer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module framer #(
   parameter  int unsigned  UnpackedWidth  = 1
  ,parameter  int unsigned  PackedNum      = 8
  ,parameter  int unsigned  PacketLenElems = 1024 // Number of packed elements per packet
  ,localparam int unsigned  PackedWidth    = UnpackedWidth * PackedNum

  ,parameter  logic [PackedWidth-1:0] TailByte0  = PackedWidth'($unsigned(165)) // 0xA5
  ,parameter  logic [PackedWidth-1:0] TailByte1  = PackedWidth'($unsigned(90))  // 0x5A
  ,parameter  logic [PackedWidth-1:0] WakeupCmd  = PackedWidth'($unsigned(153))  // 0x99
  ,localparam int unsigned            CountWidth = $clog2(PacketLenElems)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0]               valid_i
  ,output [0:0]               ready_o
  ,input  [UnpackedWidth-1:0] unpacked_i

  ,output [0:0]             valid_o
  ,input  [0:0]             ready_i
  ,output [PackedWidth-1:0] data_o
);
  // FSM States
  typedef enum logic [1:0] {Wakeup, Forward, Footer0, Footer1} fsm_e;
  fsm_e state_q, state_d;

  // Packer Wires
  wire  [0:0]             pack_ready;
  wire  [0:0]             pack_valid;
  wire  [PackedWidth-1:0] packed_data;

  // Handshake Wires
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  /* ---------------------------------------- Counter Logic ---------------------------------------- */
  // Tracks completed packet elements and drives flush logic for when packet doesn't divide evenly into packed elements.
  // Counter saturates at max count and is reset on next packet.
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
    if (state_q == Footer1 && out_fire) begin
      counter_d = '0;
    // Increment counter when accepting input in forward state
    end else if (state_q == Forward && in_fire) begin
      // Saturate at max count
      if (!counter_max) counter_d = counter_q + 1'b1;
    end
  end

  /* ------------------------------------------ Flush Logic ------------------------------------------ */
  // Flush logic for when packet doesn't divide evenly into packed elements. 
  logic [0:0] rx_complete;
  
  wire  [0:0] pack_out_fire = pack_valid && ((state_q == Forward) && ready_i); // Flush just fired
  wire  [0:0] clr_flush_req = pack_out_fire && rx_complete; // Clear flush request when packer fires after receiving last input

  logic [0:0] flush_req_d, flush_req_q;

  wire  [0:0] last_input = in_fire && counter_max;

  // Flushing logic to flush packer when receiving last input
  always_ff @(posedge clk_i) begin
    if (rst_i) flush_req_q <= 1'b0;
    else       flush_req_q <= flush_req_d;
  end

  always_comb begin
    flush_req_d = flush_req_q;
    if (last_input)    flush_req_d = 1'b1; // Set flush on last input
    if (clr_flush_req) flush_req_d = 1'b0; // Clear flush on output in forward state
  end

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  logic [PackedWidth-1:0] data_q,  data_d;
  logic [0:0]             valid_q, valid_d;
  
  wire  [0:0] tx_complete = rx_complete && out_fire;

  // RX Complete logic to track when a full packet has been received
  always_ff @(posedge clk_i) begin
    if (rst_i)                               rx_complete <= 1'b0;
    else if (state_q == Footer1 && out_fire) rx_complete <= 1'b0;
    else if (last_input)                     rx_complete <= 1'b1;
  end

  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q <= Wakeup;
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
      Wakeup: begin
        if (out_fire)    state_d = Forward;
      end
      Forward: begin
        if (tx_complete) state_d = Footer0;
      end
      Footer0: begin
        if (out_fire)    state_d = Footer1;
      end
      Footer1: begin
        if (out_fire)    state_d = Forward;
      end
      default:           state_d = Wakeup;
    endcase
  end

  // Data output is either packed data or footer bytes
  assign data_o  = rst_i ? '0 : data_d;
  // Valid output unless packer is not valid in forward state
  assign valid_o = (!rst_i) && valid_d;
  // Forward data from packer until footer state, then send tail bytes
  // Only ready to accept new input data in forward state and not complete
  assign ready_o = (state_q == Forward) && pack_ready && !rx_complete;

  // Data Logic
  always_comb begin
    data_d  = data_q;  // Default hold
    valid_d = valid_q; // Default hold
    case (state_q)
      Wakeup:  begin data_d = WakeupCmd;   valid_d = !rst_i;     end
      Forward: begin data_d = packed_data; valid_d = pack_valid; end
      Footer0: begin data_d = TailByte0;   valid_d = 1'b1;       end
      Footer1: begin data_d = TailByte1;   valid_d = 1'b1;       end
      default: begin data_d = data_q;      valid_d = valid_q;    end
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
    ,.flush_i   (flush_req_q)
    ,.valid_i   (in_fire)
    ,.ready_o   (pack_ready)

    ,.packed_o  (packed_data)
    ,.valid_o   (pack_valid)
    ,.ready_i   ((state_q == Forward) ? ready_i : 1'b0) // Only accept output in forward state
  );

endmodule
