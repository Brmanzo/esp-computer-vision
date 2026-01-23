// ram_1r1w_sync.sv
// Bradley Manzo

`timescale 1ns / 1ps
module ram_1r1w_sync #(
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

  `ifndef SYNTHESIS // YOSYS ignores
  initial begin
    for (int i = 0; i < Depth; i++) begin
      $dumpvars(0, mem[i]);
    end
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
