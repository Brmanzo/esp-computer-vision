// skid_buffer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
/* verilator lint_off PINCONNECTEMPTY */
module skid_buffer #(
   parameter  int unsigned Width    = 8
  ,parameter  int unsigned Depth    = 16
  ,parameter  int unsigned HeadRoom = 4 // Assert RTS when only 4 slots remain
  ,localparam int unsigned AddrWidth = $clog2(Depth)
)(
    input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [Width-1:0] data_i
  ,input  [0:0]       valid_i
  ,output [0:0]       ready_o

  ,output [Width-1:0] data_o
  ,output [0:0]       valid_o
  ,input  [0:0]       ready_i

  // Flow Control
  ,output [0:0] rts_o // Stop UART if high
);
  /* ------------------------------------ Pointer Declarations ------------------------------------ */
  logic [AddrWidth-1:0] read_ptr_q, read_ptr_d, write_ptr;
  logic [AddrWidth-1:0] prev_write_ptr = '0;
  logic [0:0] read_wrap, write_wrap;

  /* ------------------------------------ Full/Empty Logic ------------------------------------ */
  always_ff @ (posedge clk_i) begin
    if (rst_i) prev_write_ptr <= '0;
    else       prev_write_ptr <= write_ptr;
  end

  wire [0:0] ptr_overlap = (write_ptr == read_ptr_q);
  wire [0:0] ptr_wrap    = (write_wrap != read_wrap);

  wire [0:0] empty_w     = ptr_overlap && ~ptr_wrap;
  wire [0:0] full_w      = ptr_overlap &&  ptr_wrap;

  assign ready_o = ~full_w;
  assign valid_o = ~empty_w;

  wire [0:0] in_fire = ready_o && valid_i;
  wire [0:0] out_fire = ready_i && valid_o;

  wire [AddrWidth:0] occupancy_w;
  assign occupancy_w = {write_wrap, write_ptr} - {read_wrap, read_ptr_q};
  assign rts_o = (occupancy_w >= (AddrWidth + 1)'(Depth - HeadRoom));
  /* ------------------------------------ Bypass Logic ------------------------------------ */
  logic [Width-1:0] data_bypass;

  // Elastic Head to latch input for bypass logic
  elastic #(
     .Width(Width)
  ) elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    ,.data_i (data_i)
    ,.valid_i(valid_i)
    ,.ready_o()
    ,.valid_o()
    ,.data_o (data_bypass)
    ,.ready_i(ready_i)
  );

  wire  [Width-1:0] ram_data;
  logic [Width-1:0] data;
  assign data_o = data;

  wire [0:0] early_read = prev_write_ptr == read_ptr_q;
  wire [0:0] bypass     = early_read     && ~full_w;

  always_comb begin
    if (bypass) data = data_bypass;
    else        data = ram_data;
  end
  /* ------------------------------- Pointer Counter Instantiation ------------------------------- */
  // Read Counter
  counter #(
     .Width   (AddrWidth+1) // One extra bit for full/empty distinction
    ,.ResetVal('0)
  ) read_counter_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)
    ,.up_i   (out_fire) // Increment if ready_i and valid_o
    ,.down_i (1'b0) // Only increments then rolls over
    ,.count_o({read_wrap, read_ptr_q})
    ,.next_count_o(read_ptr_d)
  );

  // Write Counter
  counter #(
     .Width   (AddrWidth+1) // One extra bit for full/empty distinction
    ,.ResetVal(1'b0)
  ) write_counter_inst (
     .clk_i   (clk_i)
    ,.rst_i   (rst_i)
    ,.up_i    (in_fire) // Increment if ready_o and valid_i
    ,.down_i  (1'b0) // Only increments then rolls over
    ,.count_o ({write_wrap, write_ptr})
    ,.next_count_o()
  );

  /* ------------------------------------ RAM Instantiation ------------------------------------ */
  // FIFO RAM
  ram_1r1w_sync #(
    .Width(Width) // Width of FIFO specified by Width
   ,.Depth(Depth) // Depth of FIFO specified by Depth
  ) ram_1r1w_sync_inst (
     .clk_i(clk_i)
    ,.rst_i(rst_i)

    ,.wr_valid_i(in_fire) // Disable write when bypassing
    ,.wr_data_i (data_i)
    ,.wr_addr_i (write_ptr)

    ,.rd_valid_i(1'b1)
    ,.rd_addr_i (read_ptr_d)
    ,.rd_data_o (ram_data)
  );

endmodule
