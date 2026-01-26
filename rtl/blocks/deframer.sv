`timescale 1ns / 1ps

module deframer #(
   parameter  int unsigned UnpackedWidth  = 1
  ,parameter  int unsigned PackedNum      = 8
  ,parameter  int unsigned PackedWidth    = UnpackedWidth * PackedNum
  ,parameter  int unsigned PacketLenBytes = 1024 // Number of packed elements per packet
  ,localparam int unsigned PacketLenElems = PacketLenBytes * PackedNum
  ,localparam int unsigned CountWidth     = $clog2(PacketLenElems)
  
  ,parameter logic [PackedWidth-1:0] HeaderByte0 = PackedWidth'($unsigned(165)) // 0xA5
  ,parameter logic [PackedWidth-1:0] HeaderByte1 = PackedWidth'($unsigned(90))  // 0x5A
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0]               valid_i
  ,output [0:0]               ready_o
  ,input  [PackedWidth-1:0]   data_i

  ,output [0:0]               valid_o
  ,input  [0:0]               ready_i
  ,output [UnpackedWidth-1:0] unpacked_o
);
  // FSM States
  typedef enum logic [1:0] {Header0, Header1, Forward} fsm_e;
  fsm_e state_q, state_d;

  // Unpacker Wires
  wire  [0:0]               unpack_ready;
  wire  [0:0]               unpack_valid;
  wire  [UnpackedWidth-1:0] unpack_data;

  // Handshake Wires
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  /* ---------------------------------------- Counter Logic ---------------------------------------- */
  logic [CountWidth-1:0] counter_q, counter_d;

  wire  [CountWidth-1:0] max_count_w   = CountWidth'(PacketLenElems - 1);
  wire  [0:0]            counter_max_w = (counter_q == max_count_w);

  // Saturating counter to track number of packed inputs
  always_ff @(posedge clk_i) begin
    if (rst_i) counter_q <= '0;
    else       counter_q <= counter_d;
  end

  always_comb begin
    counter_d = counter_q; // Default hold
    // Reset counter when finishing footer
    if (state_q == Header0) begin
      counter_d = '0;
    // Increment counter when accepting input in forward state
    end else if ((state_q == Forward || state_q == Header1) && out_fire) begin
      // Saturate at max count
      if (!counter_max_w) counter_d = counter_q + CountWidth'(1);
      else                counter_d = counter_q;
    end
  end

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  wire  [0:0] last_input = out_fire && counter_max_w;
  logic [0:0] flushing;

  logic [0:0] rx_complete;
  wire  [0:0] tx_complete = rx_complete && out_fire;

  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q <= Header0;
      unpacked_q  <= '0;
      valid_q <= 1'b0;
    end else begin
      unpacked_q  <= unpacked_d;
      valid_q <= valid_d;
      // Advance state while filtering and forwarding
      if (in_fire  || last_input) begin
        state_q <= state_d;
      end
    end
  end

  // Next state logic
  always_comb begin
    state_d = state_q;
    case (state_q)
      Header0: begin
        if (in_fire) begin
          if (data_i == HeaderByte0) state_d = Header1;
          else                       state_d = Header0;
        end
      end
      Header1: begin
        if (in_fire) begin
          if (data_i == HeaderByte1)      state_d = Forward;
          else if (data_i == HeaderByte0) state_d = Header1;
          else                            state_d = Header0;
        end
      end
      Forward: begin
        if (last_input) state_d = Header0;
      end
      default: state_d = Header0;
    endcase
  end

  // Output data after header bytes have been received
  logic [UnpackedWidth-1:0] unpacked_q, unpacked_d;
  assign unpacked_o = unpacked_d;
  // Valid output only when forwarding valid data
  logic [0:0] valid_q, valid_d;
  assign valid_o = valid_d;
  // Forward data from packer until footer state, then send tail bytes
  // Only ready to accept new input data in forward state and not complete
  assign ready_o = (state_q == Forward) ? unpack_ready : 1'b1;

  // Data Logic
  always_comb begin
    unpacked_d  = unpacked_q;
    valid_d = valid_q;
    case (state_q)   // Invalid data until forwarding
      Header0: begin unpacked_d = '0;          valid_d = 1'b0;       end
      Header1: begin unpacked_d = '0;          valid_d = 1'b0;       end
      Forward: begin unpacked_d = unpack_data; valid_d = unpack_valid; end
      default: begin unpacked_d = '0        ;  valid_d = 1'b0;    end
    endcase
  end

  /* ------------------------------------------ Unpacker Inst ------------------------------------------ */
  // Unpacker to unpack 4 2-bit magnitude values into each 8-bit UART input
  unpacker #(
     .UnpackedWidth(UnpackedWidth)
    ,.PackedNum    (PackedNum)
  ) unpacker_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)

    ,.packed_i  (data_i)
    ,.valid_i   ((state_q == Forward) && valid_i)
    ,.ready_o   (unpack_ready)

    ,.unpacked_o(unpack_data)
    ,.valid_o   (unpack_valid)
    ,.ready_i   ((state_q == Forward) && ready_i) // Only accept output in forward state
  );

endmodule
