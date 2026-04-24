// mac.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module avg #(
   parameter  int unsigned KernelWidth = 3
  ,parameter  int unsigned InBits      = 1
  ,localparam int unsigned OutBits     = InBits
  ,localparam int unsigned KernelArea  = KernelWidth * KernelWidth
)  (
   input  logic signed [KernelArea-1:0][InBits-1:0] window // 1D Packed Array
  ,output logic signed [OutBits-1:0] data_o
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

  localparam int unsigned AccOutBits = acc_output_width(KernelArea, InBits);
  logic signed [AccOutBits-1:0] acc;

  generate
    // If binary encoding, treat {0,1} as {-1,1}
    if (InBits == 1) begin : gen_binary_avg
      always_comb begin
        acc = '0;
        for (int i = 0; i < KernelArea; i++) begin
          if (window[i][0]) acc += 1;
          else              acc -= 1;
        end
      end
    end else begin : gen_signed_avg
      always_comb begin
        acc = '0;
        for (int i = 0; i < KernelArea; i++) begin
          acc += $signed(window[i]);
        end
      end
    end
  endgenerate

  assign data_o = OutBits'(acc >>> $clog2(KernelArea)); // Divide by KernelArea (power of 2) using right shift

endmodule
