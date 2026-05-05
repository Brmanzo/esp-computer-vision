// class_framer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module class_framer #(
   parameter  int unsigned BusBits = 8
   
  ,parameter  logic [BusBits-1:0] TailByte0  = BusBits'($unsigned(165)) // 0xA5
  ,parameter  logic [BusBits-1:0] TailByte1  = BusBits'($unsigned(90))  // 0x5A
  ,parameter  logic [BusBits-1:0] WakeupCmd  = BusBits'($unsigned(153))  // 0x99
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0]          valid_i
  ,output logic [0:0]    ready_o
  ,input  [BusBits-1:0] class_i

  ,output logic [0:0]    valid_o
  ,input  [0:0]          ready_i
  ,output logic [BusBits-1:0] uart_o
);

  // FSM States
  typedef enum logic [1:0] {Wakeup, Forward, Footer0, Footer1} fsm_e;
  fsm_e state_q, state_d;

  // Handshake Wires
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = valid_o && ready_i;

  /* ------------------------------------------- FSM Logic ------------------------------------------- */
  
  // Current state logic
  always_ff @(posedge clk_i) begin
    if (rst_i) state_q <= Wakeup;
    else       state_q <= state_d;
  end

  // Next state logic
  always_comb begin
    state_d = state_q;
    case (state_q)
      Wakeup:  if (out_fire) state_d = Forward;
      Forward: if (in_fire)  state_d = Footer0; // Single byte packet, transition on input handshake
      Footer0: if (out_fire) state_d = Footer1;
      Footer1: if (out_fire) state_d = Forward; // Ready for next classification
      default: state_d = Wakeup;
    endcase
  end

  /* --------------------------------------- Output Assignments --------------------------------------- */

  // Data and Handshake Muxing
  always_comb begin
    // Defaults
    uart_o  = class_i;
    valid_o = 1'b0;
    ready_o = 1'b0;

    case (state_q)
      Wakeup: begin
        uart_o  = WakeupCmd;
        valid_o = 1'b1;
        ready_o = 1'b0; // Don't accept data while waking up
      end

      Forward: begin
        uart_o  = class_i;
        valid_o = valid_i;
        ready_o = ready_i;
      end

      Footer0: begin
        uart_o  = TailByte0;
        valid_o = 1'b1;
        ready_o = 1'b0; // Block input while sending footers
      end

      Footer1: begin
        uart_o  = TailByte1;
        valid_o = 1'b1;
        ready_o = 1'b0;
      end
    endcase
  end

endmodule
