// multi_delay_buffer
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module multi_delay_buffer #(
   parameter  int unsigned BufferWidth   = 8
  ,parameter  int unsigned Delay         = 8
  ,parameter  int unsigned BufferRows    = 2
  ,parameter  int unsigned InputChannels = 2
  ,localparam int unsigned AddrWidth     = $clog2(Delay)
  ,localparam int unsigned ChannelWidth  = BufferWidth * BufferRows
  ,localparam int unsigned RamWidth      = InputChannels * ChannelWidth
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [InputChannels-1:0][BufferWidth-1:0] data_i
  ,input  [0:0]       valid_i
  ,output [0:0]       ready_o

  ,output [InputChannels-1:0][BufferRows-1:0][BufferWidth-1:0] data_o

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
);
  /* -------------------------------------- Handshaking Logic -------------------------------------- */
  assign valid_o = valid_i;
  assign ready_o = ready_i;
  wire  [0:0] in_fire = (valid_i && ready_o);

  /* ------------------------------------ Address Counter Logic ------------------------------------ */
  // Counter rolls through RAM implementing a circular bufer
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

  /* ------------------------------------- Shift Register Logic ------------------------------------- */
  // Unpacked arrays representing the current and next states of each channel's shift register
  logic [InputChannels-1:0][ChannelWidth-1:0] channel_rd;
  logic [InputChannels-1:0][ChannelWidth-1:0] channel_wr;

  // Channels read from RAM are left shifted and new corresponding channel data is inserted at the LSB
  always_comb begin
    for (int ch = 0; ch < InputChannels; ch++) begin
      channel_wr[ch] = (channel_rd[ch] << BufferWidth) | {{(ChannelWidth-BufferWidth){1'b0}}, data_i[ch]};
    end
  end
  
  /* ------------------------------------------- RAM Logic ------------------------------------------- */ 
  // Packed arrays representing the combined channel data to be written to and read from the RAM
  logic [RamWidth-1:0] ram_wr;
  logic [RamWidth-1:0] ram_rd;

  // Pack single RAM write from unpacked channel writes
  for (genvar ch = 0; ch < InputChannels; ch++) begin : gen_pack
    assign ram_wr[ch*ChannelWidth +: ChannelWidth] = channel_wr[ch];
  end

  // Unpack from channel read from packed RAM read
  for (genvar ch = 0; ch < InputChannels; ch++) begin : gen_unpack
    assign channel_rd[ch] = ram_rd[ch*ChannelWidth +: ChannelWidth];
  end

  // Channel delay buffers vertically partitioned within shift register
  ram_1r1w_sync #(
     .Width(RamWidth)
    ,.Depth(Delay)
  ) ram_inst (
     .clk_i     (clk_i)
    ,.rst_i     (rst_i)
    ,.wr_valid_i(in_fire)
    ,.wr_data_i (ram_wr)
    ,.wr_addr_i (read_ptr_r)
    ,.rd_valid_i(in_fire)
    ,.rd_addr_i (read_ptr_r)
    ,.rd_data_o (ram_rd)
  );

  // Assign data off of RAM read to output buses
  generate
    for (genvar ch = 0; ch < InputChannels; ch++) begin : gen_out_ch
      for (genvar r = 0; r < BufferRows; r++) begin : gen_data_o
        assign data_o[ch][r] = channel_rd[ch][(r+1)*BufferWidth-1 -: BufferWidth];
      end
    end
  endgenerate

endmodule
