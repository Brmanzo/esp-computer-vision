// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module avg #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned WidthIn     = 1
  ,localparam int unsigned WidthOut    = WidthIn
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic [KernelArea-1:0][WidthIn-1:0] window // 1D Packed Array
  ,output logic [WidthOut-1:0] data_o
);

  function automatic int unsigned acc_output_width;
    input int unsigned kernel_area, quantized_width;
    longint unsigned max_input, worst_case_sum;
    begin
      max_input      = (64'd1 << quantized_width) - 1;
      worst_case_sum = longint'(kernel_area) * max_input;

      acc_output_width = $clog2(worst_case_sum + 1); // assign to function name
    end
  endfunction

  localparam int unsigned AccOutWidth = acc_output_width(KernelArea, WidthIn);
  logic signed [AccOutWidth-1:0] acc;

  always_comb begin
    acc = '0;
    for (int i = 0; i < KernelArea; i++) begin
      acc += window[i];
    end
  end
  assign data_o = acc >> $clog2(KernelArea); // Divide by KernelArea (power of 2) using right shift

endmodule
