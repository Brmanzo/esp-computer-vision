// multi_delay_ram.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module multi_delay_ram #(
   parameter int unsigned BufferCount    = 2
  ,parameter int unsigned ChannelsPerRam = 4 
  ,parameter int unsigned InBits         = 1
  ,parameter int unsigned InChannels     = 9
  ,parameter int unsigned KernelWidth    = 3
  ,parameter int unsigned LineWidthPx    = 16
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i
  
  ,input  [0:0] in_fire
  ,input  [InChannels-1:0][InBits-1:0] data_i
  
  ,output [InChannels-1:0][KernelWidth-1:1][InBits-1:0] data_o
);

  // Targetting IceStorm's 30 4kB embedded block RAMs
  // https://www.mouser.com/datasheet/2/225/iCE40%20UltraPlus%20Family%20Data%20Sheet-1149905.pdf?srsltid=AfmBOoojsqUL7qv64GuzD_fsFp6UalE__EO5sBNN2KRE01qaez2zv7uA#page=15
  // 256 x 16, 512 x 8, 1024 x 4, or 2,048 x 2 bit configurations are possible with the 4kB RAMs

  // If buffer length exceeds 256, target 8 bit wide RAM
  //    max parameters for 1 channel:    3x3 kernel with 4 bit inputs or 9x9 kernel with 1 bit inputs
  //    max channels: 4 with parameters: 3x3 kernel with 1 bit inputs
  // If buffer length is 255 or less, target 16 bit wide RAM
  //    max parameters for 1 channel:    3x3 kernel with 8 bit inputs or 17x17 kernel with 1 bit inputs
  //    max channels: 8 with parameters: 3x3 kernel with 1 bit inputs

  logic [BufferCount-1:0][ChannelsPerRam-1:0][InBits-1:0] data_i_padded;
  logic [BufferCount-1:0][ChannelsPerRam-1:0][KernelWidth-1:1][InBits-1:0] data_o_padded;

  generate
    for (genvar buf_idx = 0; buf_idx < BufferCount; buf_idx++) begin : gen_ram_buffers
      // Generate the necessary number of buffers based on the input parameters and target RAM width
      localparam int unsigned FirstCh  = buf_idx * ChannelsPerRam;
      
      // Pad inputs so each RAM has a full set of channels
      for (genvar ch = 0; ch < ChannelsPerRam; ch++) begin : gen_padded_connections
        if (FirstCh + ch < InChannels) begin : gen_data_connections
          assign data_i_padded[buf_idx][ch] = data_i[FirstCh + ch];
          assign data_o[FirstCh + ch] = data_o_padded[buf_idx][ch];
        end else begin : gen_zero_connections
          assign data_i_padded[buf_idx][ch] = '0;
        end
      end

      /* verilator lint_off PINCONNECTEMPTY */
      multi_delay_buffer #(
         .BufferWidth(InBits)
        ,.Delay      (LineWidthPx - 1)
        ,.BufferRows (KernelWidth - 1)
        ,.InputChannels(ChannelsPerRam)
      ) multi_delay_buffer_inst (
         .clk_i   (clk_i)
        ,.rst_i  (rst_i)

        ,.data_i (data_i_padded[buf_idx]) // Partition input channels across buffers
        ,.valid_i(in_fire)
        ,.ready_o()

        ,.data_o (data_o_padded[buf_idx]) // Row buffers >= 1 read from delay buffer
        ,.valid_o()
        ,.ready_i(1'b1)
      );
    end
  endgenerate
endmodule
