// mag.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module mag #(
   parameter int unsigned WidthIn = 14
  ,parameter int unsigned WidthOut = WidthIn + 1
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0]                  valid_i
  ,input  signed [WidthIn - 1:0] gx_i
  ,input  signed [WidthIn - 1:0] gy_i
  ,output signed [0:0]           ready_o

  ,output [0:0]     valid_o
  ,output [WidthOut-1:0] mag_o
  ,input [0:0]      ready_i
);
  wire [0:0] elastic_ready;
  assign ready_o = elastic_ready;

  elastic
  #(.Width((WidthIn)*2)
   ,.DatapathGate(1'b1)
   ,.DatapathReset(1)
   )
  elastic_inst
  (.clk_i(clk_i)
   ,.rst_i(rst_i)
   ,.data_i({gx_i, gy_i})
   ,.valid_i(valid_i)
   ,.ready_o(elastic_ready)
   ,.valid_o(valid_o)
   ,.data_o({gx, gy})
   ,.ready_i(ready_i)
  );

  logic [WidthIn-1:0]  gx,    gy;
  logic [WidthOut-1:0] mag_d, mag_q; // One extra bit to avoid overflow on shift and add
  assign mag_o = mag_d;

  wire [0:0]          gy_greater = {1'b0, gy} >= {gx, 1'b0};
  wire [0:0]          gx_greater = {1'b0, gx} >= {gy, 1'b0};
  wire [WidthOut-1:0] half_gx    = {1'b0, (gx >> 1)};

  always_ff @(posedge clk_i) begin
    if (rst_i) mag_q <= '0;
    else mag_q <= mag_d;
  end

  always_comb begin
    mag_d = mag_q; // Default hold
    if      (gy_greater) mag_d = {1'b0, gy};
    else if (gx_greater) mag_d = {1'b0, gx};
    // If both multiply x by 1.5 (approx of sqrt)
    else mag_d = {1'b0,gy} + half_gx;
  end

endmodule
