// avg.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module avg #(
   parameter  int unsigned KernelWidth = 2
  ,parameter  int unsigned InBits      = 1
  ,parameter  int unsigned Unsigned    = 0
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
      max_input      = (64'd1 << (quantized_width > 1 ? quantized_width : 1)) - 1;
      worst_case_sum = longint'(kernel_area) * max_input;

      acc_output_width = $clog2(worst_case_sum + 1) + 1; // +1 for sign bit
    end
  endfunction

  localparam int unsigned AccOutBits = acc_output_width(KernelArea, InBits);
  logic signed [AccOutBits-1:0] acc;

  generate
    // If binary encoding, treat {0,1} as {-1,1}
    if (InBits == 1) begin : gen_binary_avg
      always_comb begin
        // Sum of {-1, 1} encoding: count_of_1s - count_of_0s
        // count_of_0s = KernelArea - count_of_1s
        // Sum = count_of_1s - (KernelArea - count_of_1s) = 2 * count_of_1s - KernelArea
        acc = AccOutBits'(2 * $countones(window) - KernelArea);
      end
    end else begin : gen_signed_avg
      always_comb begin
        acc = '0;
        for (int i = 0; i < KernelArea; i++) begin
          if (Unsigned == 1) begin
            acc = acc + AccOutBits'($unsigned(window[i]));
          end else begin
            acc = acc + AccOutBits'($signed(window[i]));
          end
        end
      end
    end
  endgenerate

  // Efficient shift-right division for power-of-2 kernel areas
  assign data_o = OutBits'(acc >>> $clog2(KernelArea));

endmodule
