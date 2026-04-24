// max.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module max #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned InBits      = 1
  ,localparam int unsigned OutBits     = InBits
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic signed [KernelArea-1:0][InBits-1:0] window // 1D Packed Array
  ,output logic signed [OutBits-1:0] data_o
);
  logic signed [OutBits-1:0] max;

  generate
    if (InBits == 1) begin : gen_binary_max
      always_comb begin
        max = |window;
      end
    end else begin : gen_signed_max
      always_comb begin
        max = window[0];
        for (int i = 1; i < KernelArea; i++) begin
          if (window[i] > max) max = window[i];
        end
      end
    end
  endgenerate
  assign data_o = max;

endmodule
