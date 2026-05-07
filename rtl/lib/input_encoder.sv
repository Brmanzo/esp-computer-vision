// output_encoder
// Bradley Manzo 2026

`timescale 1ns / 1ps

module input_encoder #(
   parameter int unsigned InBits 
  ,parameter int unsigned OutBits
)  (
   input  logic signed [InBits-1:0]  data_i
  ,output logic signed [OutBits-1:0] data_o
);
  generate
    if (InBits == 1) begin: gen_binary_in
      assign data_o = (data_i[0]) ? OutBits'( 1) : 
                                    OutBits'(-1);
    end else if (InBits == 2) begin : gen_ternary_in
      assign data_o = (data_i == 2'sb01) ?   OutBits'( 1) :
                      (data_i == 2'sb11) ? - OutBits'( 1) :
                                             OutBits'( 0);
    end else begin : gen_full_in
      assign data_o = OutBits'($signed(data_i));
    end
  endgenerate 
endmodule
