`timescale 1ns / 1ps
module packer #(
   parameter  int unsigned UnpackedWidth = 2
  ,parameter  int unsigned PackedNum     = 4
  ,parameter  int unsigned PackedWidth   = UnpackedWidth * PackedNum
  ,localparam int unsigned CountWidth    = $clog2(PackedNum)
  ,localparam int unsigned OffsetWidth   = $clog2(PackedWidth)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [UnpackedWidth-1:0] unpacked_i
  ,input  [0:0]               flush_i
  ,input  [0:0]               valid_i
  ,output [0:0]               ready_o

  ,output [PackedWidth-1:0] packed_o
  ,output [0:0]             valid_o
  ,input  [0:0]             ready_i
);

  /*  ------------------------------------ Flush Logic ------------------------------------ */
  wire  [0:0] elastic_ready;
  wire  [0:0] last       = (counter_q == max_count);
  wire  [0:0] partial    = (counter_q != '0);

  wire  [0:0] in_fire    = valid_i && ready_o;
  // Block upstream data while receiving fourth input or while flushing
  assign ready_o         = (last || flush_i) ? elastic_ready : 1'b1;

  // Determine if flush is occurring with input
  wire  [0:0] flush_partial = flush_i && partial && !in_fire;
  wire  [0:0] flush_in      = flush_i && in_fire;

  // Output when either completing final pack, when flush on input, or when flushing without input
  // flush_i is decoupled from valid_i, so we can flush even if no input is valid
  // If flush_i asserted multiple cycles in a row, we want to bypass valid data to the output without packing.
  wire [0:0] out_fire = (last && in_fire) || flush_in ||
                        (flush_partial && elastic_ready);

  /* ------------------------------------ Counter Logic ------------------------------------ */
  logic [CountWidth-1:0] counter_q, counter_d;
  wire  [CountWidth-1:0] max_count = CountWidth'(PackedNum - 1);

  always_ff @(posedge clk_i) begin
    if (rst_i) counter_q <= '0;
    else counter_q <= counter_d;
  end

  always_comb  begin
    counter_d = counter_q; // Default
    // Clear counter on proper pack or flush
    if (out_fire)     counter_d = '0;
    // Increment counter on accepted input
    else if (in_fire) counter_d = counter_q + 1;
  end

  /* ------------------------------------ Packing Logic ------------------------------------ */
  logic [PackedWidth-1:0] packed_q, packed_d;

  always_ff @(posedge clk_i) begin
      if (rst_i)         packed_q <= '0;
      // We just sent the full word to elastic; clear buffer for next frame
      else if (out_fire) packed_q <= '0;
      // We accepted a partial chunk; accumulate it
      else if (in_fire)  packed_q <= packed_d;
    end

  // Maintain offset of current shift/select within shift_reg
  logic [OffsetWidth-1:0] offset;
  logic [PackedWidth-1:0] shift_reg;

  // Apply shift and accumulate
  always_comb begin
    offset    = OffsetWidth'(counter_q * UnpackedWidth);
    shift_reg = PackedWidth'(unpacked_i);
    packed_d  = packed_q | (shift_reg << offset);
  end

  /* ------------------------------------ Elastic Interface ------------------------------------ */
  wire [PackedWidth-1:0] elastic_data = flush_partial ? packed_q : packed_d;

  elastic #(
     .Width        (PackedWidth)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    ,.data_i (elastic_data)
    ,.valid_i(out_fire)
    ,.ready_o(elastic_ready)
    ,.valid_o(valid_o)
    ,.data_o (packed_o)
    ,.ready_i(ready_i)
  );

endmodule
