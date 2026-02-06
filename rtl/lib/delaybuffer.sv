// delaybuffer.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module delaybuffer #(
   parameter  int unsigned Width     = 8
  ,parameter  int unsigned Delay     = 8
  ,localparam int unsigned AddrWidth = $clog2(Delay)
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [Width-1:0] data_i
  ,input  [0:0]       valid_i
  ,output [0:0]       ready_o

  ,output [0:0]       valid_o
  ,output [Width-1:0] data_o
  ,input  [0:0]       ready_i
);

  logic [AddrWidth-1:0] read_ptr_r;
  logic [Width-1:0]     elastic_data_d, elastic_data_q;

  wire [0:0] in_fire = (valid_i && ready_o);

  elastic #(
     .Width(Width)
    ,.DatapathGate(1'b1)
  ) elastic_head_inst (
    .clk_i   (clk_i)
    ,.rst_i  (rst_i)
    ,.data_i (data_i)
    ,.valid_i(valid_i)
    ,.ready_o(ready_o)
    ,.valid_o(valid_o)
    ,.data_o (elastic_data_d) // When valid data, put onto shift register
    ,.ready_i(ready_i)
  );

  // Register output of elastic_data
  always_ff @(posedge clk_i) begin
    if (rst_i)        elastic_data_q <= '0;
    else if (in_fire) elastic_data_q <= elastic_data_d;
  end

  counter_roll #(
     .Width(AddrWidth)
    ,.ResetVal('0)
  ) read_counter_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    ,.max_val_i(AddrWidth'(Delay-1))
    ,.up_i     (in_fire)
    ,.down_i   (1'b0)
    ,.count_o  (read_ptr_r)
  );

  logic [Width-1:0] ram_data;

  ram_1r1w_sync #(
     .Width(Width)
    ,.Depth(Delay)
  ) ram_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    ,.wr_valid_i(in_fire)
    ,.wr_data_i (data_i)
    ,.wr_addr_i (read_ptr_r)
    ,.rd_valid_i(in_fire)
    ,.rd_addr_i (read_ptr_r)
    ,.rd_data_o (ram_data)
  );

  assign data_o = (Delay == 1) ? elastic_data_q : ram_data;
endmodule
