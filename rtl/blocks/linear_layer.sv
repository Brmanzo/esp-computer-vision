// linear_layer.sv
// Bradley Manzo, 2026

module linear_layer #(
   parameter int unsigned InBits      = 1
  ,parameter int unsigned OutBits     = 1
  ,parameter int unsigned WeightBits  = 2
  ,parameter int unsigned BiasBits    = 8
  ,parameter int unsigned InChannels   = 1
  ,parameter int unsigned OutChannels  = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightBits
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasBits-1:0]   Biases  = '0
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  logic signed [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutChannels-1:0][OutBits-1:0] data_o
);

  /* ------------------------------------ Elastic Handshaking Logic ------------------------------------ */
  // Provided Elastic State Machine Logic
  logic [0:0] valid_r;
  wire  [0:0] in_fire = valid_i && ready_o;

  always_ff @(posedge clk_i) begin
    if (rst_i)        valid_r <= 1'b0;
    else if (ready_o) valid_r <= in_fire;
  end

  assign valid_o =  valid_r;
  assign ready_o = ~valid_r | ready_i;

  /* --------------------------------------- Output Channel Logic --------------------------------------- */
  // Each output channel is a neuron with the same input data but different weights and biases

  logic signed [InChannels-1:0][InBits-1:0] data_q;

  always_ff @(posedge clk_i) begin
    if (rst_i) data_q <= '0;
    else if (in_fire) data_q <= data_i;
  end

  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
      neuron #(
         .InBits    (InBits)
        ,.OutBits   (OutBits)
        ,.WeightBits(WeightBits)
        ,.BiasBits  (BiasBits)
        ,.InChannels (InChannels)

        ,.Weights(Weights[ch*WeightIndex +: WeightIndex])
        ,.Bias   (Biases[ch*BiasBits +: BiasBits])
      ) neuron_inst (
         .data_i(data_q)
        ,.data_o(data_o[ch])
      );
    end
  endgenerate

endmodule
