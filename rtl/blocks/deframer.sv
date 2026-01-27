`timescale 1ns / 1ps

module deframer #(
   parameter  int unsigned UnpackedWidth      = 1
  ,parameter  int unsigned PackedNum          = 8
  ,localparam int unsigned PackedWidth        = UnpackedWidth * PackedNum
  ,parameter  int unsigned PacketLenElems     = 1024 // Number of packed elements per packet
  ,localparam int unsigned CountWidth         = $clog2(PacketLenElems)
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

  wire  [0:0] counter_max = (counter_q == MaxCount);

  counter #(
    .Width(CountWidth)
  )  elem_counter_inst (
    .clk_i(clk_i)
    ,.rst_i(rst_i || (state_q == Header0))

    ,.up_i((state_q == Forward) && in_fire && !counter_max)
    ,.down_i(1'b0)

    ,.count_o(counter_q)
    ,.next_count_o()
  );

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  wire  [0:0] last_output = (state_q == Forward) && out_fire && unpack_done && counter_max;
  logic [0:0] last_output_q;
  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q <= Header0;
      unpacked_q  <= '0;
      valid_q <= 1'b0;
      last_output_q <= 1'b0;
    end else begin
      unpacked_q  <= unpacked_d;
      valid_q <= valid_d;
      // Latch last_output to end forwarding state when ready_o
      if (state_q == Forward && last_output) last_output_q <= 1'b1;
      if (state_q == Header0)                last_output_q <= 1'b0;
      // Advance state while filtering and forwarding
      if (((state_q != Forward) && in_fire) || last_output_q) begin
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
        if (last_output_q && ready_i) state_d = Header0;
      end
      default: state_d = Header0;
    endcase
  end

  // Data Logic
  logic [UnpackedWidth-1:0] unpacked_q, unpacked_d;
  logic [0:0] valid_q, valid_d;

  // Output data after header bytes have been received
  always_comb begin
    unpacked_d  = unpacked_q;
    valid_d = valid_q;
    case (state_q)   // Invalid data until forwarding
      Header0: begin unpacked_d = '0;          valid_d = 1'b0;         end
      Header1: begin unpacked_d = '0;          valid_d = 1'b0;         end
      Forward: begin unpacked_d = unpack_data; valid_d = unpack_valid; end
      default: begin unpacked_d = '0;          valid_d = 1'b0;         end
    endcase
  end
  /* --------------------------------------- Output Assignments --------------------------------------- */
  // Assigning combinational data to output in same cycle as unpacker
  assign unpacked_o = unpacked_d;
  // Valid output only when forwarding valid data
  assign valid_o = valid_d;
  // Always ready for data during header states, when forwarding,
  // we defer forward pressure to the unpacker module
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
    ,.valid_i   ((state_q == Forward) && valid_i && !counter_max) // Inputs are only valid in forward state
    ,.ready_o   (unpack_ready)

    ,.unpacked_o(unpack_data)
    ,.valid_o   (unpack_valid)
    ,.ready_i   ((state_q == Forward) && ready_i) // Only accept output in forward state
    ,.done_o    (unpack_done)
  );

endmodule
