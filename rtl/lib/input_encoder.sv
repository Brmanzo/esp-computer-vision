// input_encoder.sv
// Bradley Manzo 2026

`timescale 1ns / 1ps

module input_encoder #(
   parameter int unsigned Unsigned = 1  // 1: zero-extend (post-ReLU unsigned), 0: sign-extend
  ,parameter int unsigned InBits
  ,parameter int unsigned OutBits
)  (
   input  logic signed [InBits-1:0]  data_i
  ,output logic signed [OutBits-1:0] data_o
);
  generate
    if (InBits == 1) begin: gen_binary_in
      assign data_o = (data_i[0]) ? OutBits'(1) : 
                                   -OutBits'(1);
    end else if (Unsigned == 1) begin : gen_unsigned_in
      assign data_o = OutBits'({1'b0, data_i});  // zero-extend: MSB=0, always positive
    end else begin : gen_full_in
      assign data_o = OutBits'($signed(data_i));
    end
  endgenerate 
endmodule
