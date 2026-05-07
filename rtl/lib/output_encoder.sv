// output_encoder
// Bradley Manzo 2026

`timescale 1ns / 1ps

module output_encoder #(
   parameter int unsigned InBits 
  ,parameter int unsigned OutBits
)  (
   input  logic signed [InBits-1:0]  data_i
  ,output logic signed [OutBits-1:0] data_o
);
  generate
    // Binary Output Encoding {-1,1} -> {0,1}
    if (OutBits == 1) begin : gen_binary_out
      assign data_o = (data_i > 0) ? OutBits'(1) : OutBits'(0);
    // Ternary Output Encoding {-1,0,1}
    end else if (OutBits == 2) begin : gen_ternary_out
      assign data_o = (data_i > 0) ? OutBits'(1) :
                      (data_i < 0) ? OutBits'(-1) :
                                     OutBits'(0);
    // Full or Extended Output (No truncation)
    end else if (OutBits >= InBits) begin : gen_full_out
      assign data_o = OutBits'($signed(data_i));
    // Truncated MSB Output (Shift-Right)
    end else begin : gen_truncated_out
      assign data_o = data_i[InBits-1 -: OutBits];
    end
  endgenerate
endmodule
