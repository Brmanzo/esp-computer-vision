// multi_delay_buffer
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module multi_delay_buffer #(
   parameter  int unsigned Width       = 8
  ,parameter  int unsigned Delay       = 8
  ,parameter  int unsigned BufferCnt   = 1
  ,localparam int unsigned AddrWidth   = $clog2(Delay)
  ,localparam int unsigned OutputWidth = Width * BufferCnt
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [Width-1:0] data_i
  ,input  [0:0]       valid_i
  ,output [0:0]       ready_o

  ,output [BufferCnt-1:0][Width-1:0] data_o
  ,output [0:0]                      valid_o
  ,input  [0:0]                      ready_i
);
  // Handshaking
  assign valid_o = valid_i;
  assign ready_o = ready_i;
  wire  [0:0] in_fire = (valid_i && ready_o);

  // Counter to roll through RAM of delay depth
  logic [AddrWidth-1:0] read_ptr_r;

  counter_roll #(
     .Width(AddrWidth)
    ,.ResetVal('0)
  ) read_counter_inst (
     .clk_i    (clk_i)
    ,.rst_i    (rst_i)
    ,.max_val_i(AddrWidth'(Delay-1)) // Delay offset by single cycle sync-RAM rd-delay
    ,.up_i     (in_fire)
    ,.down_i   (1'b0)
    ,.count_o  (read_ptr_r)
  );

  // Every word written to RAM is data_i shifted onto the data read out of RAM
  logic [OutputWidth-1:0] shift_reg;
  assign shift_reg = {{(OutputWidth-Width){1'b0}}, data_i};

  // Combinationally calculated for single cycle delay
  logic [OutputWidth-1:0] ram_q;
  logic [OutputWidth-1:0] wr_word;
  always_comb begin
    wr_word = (ram_q << Width) | shift_reg;
  end

  // Parallel delay buffers are vertically partitioned within singular ram_1r1w_sync
  ram_1r1w_sync #(
     .Width(OutputWidth)
    ,.Depth(Delay)
  ) ram_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    ,.wr_valid_i(in_fire)
    ,.wr_data_i (wr_word)
    ,.wr_addr_i (read_ptr_r)
    ,.rd_valid_i(in_fire)
    ,.rd_addr_i (read_ptr_r)
    ,.rd_data_o (ram_q)
  );
  // Assign data off of RAM read to output buses
  generate
    genvar i;
    for (i = 0; i < BufferCnt; i++) begin : gen_data_o
      assign data_o[i] = ram_q[(i+1)*Width-1 -: Width];
    end
  endgenerate

endmodule
