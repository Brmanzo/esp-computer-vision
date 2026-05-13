// output_encoder
// Bradley Manzo 2026

`timescale 1ns / 1ps

module output_encoder #(
   parameter int unsigned InBits 
  ,parameter int unsigned OutBits
  ,parameter int unsigned ShiftBits = 0
)  (
   input  logic signed [InBits-1:0]  data_i
  ,output logic signed [OutBits-1:0] data_o
);
  // Full unsigned max: 2^OutBits - 1.  Safe because neuron_seq zero-extends
  // this port before multiplying, treating it as unsigned [0, 2^OutBits - 1].

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
    // Learned Shift Output: barrel-shift selects bit window, ReLU clamps negatives,
    // saturates at signed max so downstream signed consumers never see a negative
    // saturated value (e.g. 4'b1111 would be -1 signed, not +15).
    end else begin : gen_learned_shift
      logic [InBits-1:0] shifted;
      assign shifted = data_i >> ShiftBits;  // logical shift: fills 0s from MSB
      assign data_o = (data_i < 0)                ? '0              :  // ReLU: clamp negatives
                      (shifted > {OutBits{1'b1}}) ? {OutBits{1'b1}} :  // Saturate at unsigned max
                      OutBits'(shifted);                               // Pass shifted slice
    end
  endgenerate
endmodule
