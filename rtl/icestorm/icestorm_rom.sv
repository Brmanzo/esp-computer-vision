// icestorm_rom.sv
// Bradley Manzo 2026

// Targetting IceStorm's 30 4kB embedded block RAMs
// https://www.mouser.com/datasheet/2/225/iCE40%20UltraPlus%20Family%20Data%20Sheet-1149905.pdf?srsltid=AfmBOoojsqUL7qv64GuzD_fsFp6UalE__EO5sBNN2KRE01qaez2zv7uA#page=15
`timescale 1ns / 1ps

module icestorm_rom #(
   parameter  int unsigned Width     = 8
  ,parameter  int unsigned Depth     = 8
  ,localparam int unsigned DepthBits = (Depth <= 1) ? 1 : $clog2(Depth)
  ,parameter  int unsigned Init      = 1
  ,parameter               FileName  = "model/data/roms/hex/zeros.hex"
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [$clog2(Depth)-1 : 0] rd_addr_i
  ,output [Width-1 : 0] rd_data_o

  ,input [0:0] wr_valid_i
  ,input [Width-1 : 0] wr_data_i
  ,input [$clog2(Depth)-1 : 0] wr_addr_i
);

   // Minimum depth of 256 forces BRAM mapping; smaller arrays fall back to DFF
   // on iCE40, which never retains init data after power-on.
   localparam int unsigned ActualDepth     = (Depth < 256) ? 256 : Depth;
   localparam int unsigned ActualDepthBits = (ActualDepth <= 1) ? 1 : $clog2(ActualDepth);

   logic [Width-1:0] rom [ActualDepth-1:0];
   logic [Width-1:0]  rd_data_r;
   assign rd_data_o = rd_data_r;

   // Yosys only extracts $readmemh into BRAM INIT when it is a top-level statement
   // in an initial block — any if/generate wrapper disables extraction.
   // Default FileName points to a real zeros file so bare $readmemh never fails.
`ifdef SYNTHESIS
   initial $readmemh(FileName, rom, 0, Depth-1);
`else
   initial begin
      /* verilator lint_off WIDTHEXPAND */
      if (Init != 0 && FileName != "none" && FileName != "model/data/roms/hex/zeros.hex") begin
      /* verilator lint_on WIDTHEXPAND */
         $readmemh(FileName, rom, 0, Depth-1);
      end
      $display("%m: ROM[0] = %h, ROM[1] = %h", rom[0], rom[1]);
      for (int i = 0; i < Depth; i++)
        $dumpvars(0, rom[i]);
   end
`endif

   // Synchronous read and write
   always_ff @(posedge clk_i) begin
      rd_data_r <= rom[ActualDepthBits'(rd_addr_i)];
      // Writing during reset should not affect memory contents
      if (wr_valid_i && !rst_i) begin
         rom[ActualDepthBits'(wr_addr_i)] <= wr_data_i;
      end
   end
endmodule
