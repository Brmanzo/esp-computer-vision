// icestorm_ram.sv
// Bradley Manzo, 2026

// Targetting IceStorm's 30 4kB embedded block RAMs
// https://www.mouser.com/datasheet/2/225/iCE40%20UltraPlus%20Family%20Data%20Sheet-1149905.pdf?srsltid=AfmBOoojsqUL7qv64GuzD_fsFp6UalE__EO5sBNN2KRE01qaez2zv7uA#page=14
`timescale 1ns / 1ps
module icestorm_ram #(
   parameter  int unsigned Width     = 8
  ,parameter  int unsigned Depth     = 512
  ,localparam int unsigned AddrWidth = $clog2(Depth)
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input [0:0]           wr_valid_i
  ,input [Width-1:0]     wr_data_i
  ,input [AddrWidth-1:0] wr_addr_i

  ,input  [0:0]           rd_valid_i
  ,input  [AddrWidth-1:0] rd_addr_i
  ,output [Width-1:0]     rd_data_o
);

  logic [Width-1:0] mem [Depth];
  // Outputs registered data instead of registering address
  logic [Width-1:0]  rd_data_r;
  // Assign read output synchronously
  assign rd_data_o = rd_data_r;

  `ifndef SYNTHESIS
    initial begin
      for (int unsigned i = 0; i < Depth; i++) begin
        mem[i] = '0;
      end
      rd_data_r = '0;
    end
  `endif

  // Synchronous read and write
  always_ff @(posedge clk_i) begin
    if (~rst_i) begin
      if (rd_valid_i) rd_data_r <= mem[rd_addr_i]; // registers current memory to be output
      if (wr_valid_i) mem[wr_addr_i] <= wr_data_i;
    end
  end

endmodule
