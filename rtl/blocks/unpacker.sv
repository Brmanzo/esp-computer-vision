`timescale 1ns / 1ps
module unpacker #(
   parameter  int unsigned  UnpackedWidth = 2
  ,parameter  int unsigned  PackedNum     = 4
  ,parameter  int unsigned  PackedWidth   = UnpackedWidth * PackedNum
  ,localparam int unsigned  CountWidth    = $clog2(PackedNum)
  ,localparam int unsigned  OffsetWidth   = $clog2(PackedWidth)
  ,localparam int unsigned  ExtendWidth   = PackedWidth - UnpackedWidth
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [PackedWidth-1:0] packed_i
  ,input  [0:0]             valid_i
  ,output [0:0]             ready_o

  ,output [UnpackedWidth-1:0] unpacked_o
  ,output [0:0]               valid_o
  ,input  [0:0]               ready_i
);

  /* -------------------------- Counter Logic -------------------------- */
  wire  [CountWidth-1:0] counter;
  wire  [CountWidth-1:0] max_count = CountWidth'(PackedNum - 1);

  /* -------------------------- Unpacking Logic -------------------------- */
  logic [0:0]               unpacking;
  logic [PackedWidth-1:0]   shift_reg;
  logic [UnpackedWidth-1:0] unpacked;

  // Maintain offset of current shift/select within shift_reg
  logic [OffsetWidth-1:0] offset;

  // Mask to select the unpacked data from buffer (Replicating proper bit width)
  wire [PackedWidth-1:0] mask = {{ExtendWidth{1'b0}}, max_count};

  /* -------------------------- Handshaking Logic -------------------------- */
  wire  [0:0] elastic_ready;
  wire  [0:0] in_fire  = valid_i && ready_o;
  wire  [0:0] out_fire = unpacking && elastic_ready;
  wire  [0:0] last     = (counter == max_count);
  wire  [0:0] done     = last && out_fire;
  assign ready_o       = (~unpacking) || done;

  /* -------------------------- Current State Logic -------------------------- */
  always_ff @(posedge clk_i) begin
    if (rst_i) begin
      shift_reg <= '0;
      unpacking <= 1'b0;
    end else begin
      if (in_fire) begin
        shift_reg <= packed_i;
        unpacking <= 1'b1;
      end else if (done && !in_fire) begin
        unpacking <= 1'b0;
      end
    end
  end

  /* -------------------------- Data Path Logic -------------------------- */
  always_comb begin
    offset = OffsetWidth'(counter * UnpackedWidth);
    unpacked = UnpackedWidth'((shift_reg >> offset) & mask);
  end

  // Counter to increment unpacking offset
  counter_roll #(
     .Width   (CountWidth)
    ,.ResetVal('0)
  ) counter_roll_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    ,.max_val_i(max_count)
    ,.up_i     (out_fire)
    ,.down_i   ('0)
    ,.count_o  (counter)
  );

  // Elastic Buffer to decouple unpacking from downstream logic
  elastic #(
     .Width(UnpackedWidth)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    ,.data_i (unpacked)
    ,.valid_i(unpacking)
    ,.ready_o(elastic_ready)
    ,.valid_o(valid_o)
    ,.data_o (unpacked_o)
    ,.ready_i(ready_i)
  );

endmodule
