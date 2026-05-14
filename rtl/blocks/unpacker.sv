// unpacker.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module unpacker #(
   parameter  int unsigned  UnpackedWidth          = 2
  ,parameter  int unsigned  PackedNum              = 4
  ,parameter  int unsigned  PackedWidth            = UnpackedWidth * PackedNum
  ,localparam int unsigned  CountWidth             = $clog2(PackedNum)
  ,localparam int unsigned  ElasticWidth           = UnpackedWidth + 1
  ,localparam int unsigned  OffsetWidth            = $clog2(PackedWidth)
  ,localparam int unsigned  MaxCount               = PackedNum - 1
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [PackedWidth-1:0] packed_i
  ,input  [0:0]             valid_i
  ,output [0:0]             ready_o

  ,output [UnpackedWidth-1:0] unpacked_o
  ,output [0:0]               valid_o
  ,input  [0:0]               ready_i
  ,output [0:0]               done_o
);

  /* -------------------------- State Machine -------------------------- */
  typedef enum logic [1:0] {Idle, Unpacking, Done} fsm_e;
  fsm_e state_q, state_d;

  /* -------------------------- Datapath -------------------------- */
  logic [PackedWidth-1:0]   shift_reg;
  logic [UnpackedWidth-1:0] unpacked;
  logic [OffsetWidth-1:0]   offset;

  /* -------------------------- Handshaking Logic -------------------------- */
  wire  [0:0]             elastic_ready;

  wire  [0:0] out_fire  = (state_q == Unpacking) && elastic_ready;
  wire  [0:0] in_fire   = valid_i && ready_o;

  /* ------------------------------------ Counter Logic ------------------------------------ */
  wire  [CountWidth-1:0] counter_q;
  wire  [0:0]            last;
  
  // Counter to increment unpacking offset
  /* verilator lint_off PINCONNECTEMPTY */
  counter_roll #(
     .CountBits  (CountWidth)
    ,.MaxVal     (MaxCount)
    ,.ResetVal   (0)
    ,.EnableDown (1'b0)
  ) counter_roll_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.up_i    (out_fire)
    ,.down_i  (1'b0)
    ,.count_o (counter_q)
    ,.next_o  ()
    ,.max_o   (last)
  );
  /* verilator lint_on PINCONNECTEMPTY */

  wire [0:0] done = last && out_fire;

  assign ready_o   = (state_q != Unpacking) || done;
  /* -------------------------- State Register -------------------------- */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      state_q   <= Idle;
      shift_reg <= '0;
    end else begin
      state_q <= state_d;
      if (in_fire) shift_reg <= packed_i;
    end
  end

  /* -------------------------- Next State Logic -------------------------- */
  always_comb begin
    state_d = state_q;
    case (state_q)
      Idle: if (in_fire) state_d = Unpacking;
      Unpacking: begin
        if (done) begin
          if (in_fire) state_d = Unpacking;
          else         state_d = Done;
        end
      end
      Done: begin
        if (in_fire) state_d = Unpacking;
        else         state_d = Idle;
      end
      default: state_d = Idle;
    endcase
  end

  /* -------------------------- Data Path Logic -------------------------- */
  always_comb begin
    offset   = OffsetWidth'(counter_q * UnpackedWidth);
    unpacked = shift_reg[offset +: UnpackedWidth];
  end

  wire [ElasticWidth-1:0] elastic_out;
  assign unpacked_o = elastic_out[UnpackedWidth-1:0];
  assign done_o     = elastic_out[UnpackedWidth] && valid_o && ready_i;

  elastic #(
     .InBits       (ElasticWidth)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.data_i  ({last, unpacked})
    ,.valid_i ((state_q == Unpacking))
    ,.ready_o (elastic_ready)
    ,.valid_o (valid_o)
    ,.data_o  (elastic_out)
    ,.ready_i (ready_i)
  );

endmodule
