// deframer.sv
// Bradley Manzo, 2026
`timescale 1ns / 1ps

module deframer #(
   parameter  int unsigned UnpackedWidth      = 1
  ,parameter  int unsigned PackedNum          = 8
  ,localparam int unsigned PackedWidth        = UnpackedWidth * PackedNum
  ,parameter  int unsigned PacketLenElems     = 1024 // Number of packed elements per packet
  ,localparam int unsigned CountWidth         = $clog2(PacketLenElems + 1)
  ,localparam logic [CountWidth-1:0] MaxCount = CountWidth'(PacketLenElems)
  
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
  wire  [0:0]               unpack_done;

  // Internal Signals
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  /* ---------------------------------------- Counter Logic ---------------------------------------- */
  logic [CountWidth-1:0] counter_q;

  /* verilator lint_off PINCONNECTEMPTY */
  counter #(
    .Width(CountWidth)
  )  elem_counter_inst (
    .clk_i(clk_i)
    ,.rst_i(rst_i || (state_q == Header0))

    // Stop incrementing once we've reached the last element
    ,.up_i((state_q == Forward) && in_fire && !counter_max)
    ,.down_i(1'b0)

    ,.count_o(counter_q)
    ,.next_count_o()
  );

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  // counter_max now signals when we are processing the VERY LAST word of the packet
  wire  [0:0] counter_max = (counter_q == CountWidth'(PacketLenElems - 1));
  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q       <= Header0;
    end else begin
      // Advance state: header states on in_fire, Forward once last word is accepted
      if (((state_q != Forward) && in_fire) || (state_q == Forward && in_fire && counter_max)) begin
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
        if (in_fire && counter_max) state_d = Header0;
      end
      default: state_d = Header0;
    endcase
  end

  /* --------------------------------------- Output Assignments --------------------------------------- */
  // Output signals are driven directly by the unpacker to allow overlapping packets
  assign unpacked_o = unpack_data;
  assign valid_o    = unpack_valid;

  // Always ready for data during header states. When forwarding,
  // we defer pressure to the unpacker unless we are finishing the packet.
  assign ready_o = (state_q == Forward) ? unpack_ready : 1'b1;


  /* ------------------------------------------ Unpacker Inst ------------------------------------------ */
  // Unpacker to unpack 4 2-bit magnitude values into each 8-bit UART input
  unpacker #(
     .UnpackedWidth(UnpackedWidth)
    ,.PackedNum    (PackedNum)
  ) unpacker_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)

    ,.packed_i  (data_i)
    ,.valid_i   ((state_q == Forward) && valid_i) // Inputs are only valid in forward state
    ,.ready_o   (unpack_ready)

    ,.unpacked_o(unpack_data)
    ,.valid_o   (unpack_valid)
    ,.ready_i   (ready_i) // Allow unpacker to finish even if FSM transitions to Header0
    ,.done_o    (unpack_done)
  );

endmodule
