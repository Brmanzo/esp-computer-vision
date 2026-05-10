// icestorm_rom.sv
// Bradley Manzo 2026

// Targetting IceStorm's 30 4kB embedded block RAMs
// https://www.mouser.com/datasheet/2/225/iCE40%20UltraPlus%20Family%20Data%20Sheet-1149905.pdf?srsltid=AfmBOoojsqUL7qv64GuzD_fsFp6UalE__EO5sBNN2KRE01qaez2zv7uA#page=15
`timescale 1ns / 1ps

module icestorm_rom #(
   parameter  int unsigned Width     = 8
  ,parameter  int unsigned Depth     = 8
  ,localparam int unsigned DepthBits = $clog2(Depth)
  ,parameter  int unsigned Init      = 1
  ,parameter string        FileName  = "memory_init_file.bin"
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0]           wr_valid_i
  ,input [Width-1:0]     wr_data_i
  ,input [DepthBits-1:0] wr_addr_i

  ,input  [DepthBits-1:0] rd_addr_i
  ,output [Width-1:0]     rd_data_o
);

   logic [Width-1:0] rom [Depth-1:0];
   initial begin
      // Display depth and width (You will need to match these in your init file)
      $display("%m: Depth is %d, Width is %d", Depth, Width);
      // wire [bar:0] foo [baz:0];
      if(Init != 0) begin // if Init is 1, use readmemh.
         $readmemh(FileName, rom, 0, Depth-1);
      end
      // In order to get the memory contents in iverilog you need to run this for loop during initialization:
      // synopsys translate_off
      for (int i = 0; i < Depth; i++)
        $dumpvars(0, rom[i]);
      // synopsys translate_on
   end
   
   // Asynchronous read
   assign rd_data_o = rom[rd_addr_i];

   // Synchronous write
   always_ff @(posedge clk_i) begin
      // Writing during reset should not affect memory contents
      if (wr_valid_i && !rst_i) begin
         rom[wr_addr_i] <= wr_data_i;
      end
   end


endmodule