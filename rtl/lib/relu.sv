// relu.sv 
// Bradley Manzo, 2026

module relu #(
   parameter int unsigned InputWidth  = 1
  ,parameter int unsigned OutputWidth = InputWidth
  ,parameter int unsigned InChannels  = 1
  ,parameter int unsigned OutChannels = InChannels
)  (
   input    signed [InChannels-1:0] [InputWidth-1:0]  data_i
  ,output unsigned [OutChannels-1:0][OutputWidth-1:0] data_o
)
  // ReLU activation function: outputs 0 if input is negative, else passes through input value.
  generate
    for (genvar ch = 0; ch < InChannels; ch++) begin : gen_channels
      assign data_o[ch] = (data_i[ch][InputWidth-1]) ? '0 : data_i[ch]; // If sign bit is 1 (negative), output 0, else pass through input value.
    end
  endgenerate
endmodule
