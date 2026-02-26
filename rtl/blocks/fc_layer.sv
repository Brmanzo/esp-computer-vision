// fc_layer.sv
// Bradley Manzo, 2026

module fc_layer #(
   parameter int unsigned WidthIn      = 1
  ,parameter int unsigned WidthOut     = 1
  ,parameter int unsigned WeightWidth  = 2
  ,parameter int unsigned BiasWidth    = 8
  ,parameter int unsigned InChannels   = 1
  ,parameter int unsigned OutChannels  = 1

  ,localparam int unsigned WeightIndex = InChannels * WeightWidth
  ,parameter logic signed [OutChannels*WeightIndex-1:0] Weights = '0
  ,parameter logic signed [OutChannels*BiasWidth-1:0]   Biases  = '0
)  (
   input [0:0] clk_i
  ,input [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o
  ,input  [InChannels-1:0][WidthIn-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output logic signed [OutChannels-1:0][WidthOut-1:0] data_o
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

  logic [InChannels-1:0][WidthIn-1:0] data_q;

  always_ff @(posedge clk_i) begin
    if (rst_i) data_q <= '0;
    else if (in_fire) data_q <= data_i;
  end

  generate
    for (genvar ch = 0; ch < OutChannels; ch++) begin : gen_neurons
      neuron #(
         .WidthIn    (WidthIn)
        ,.WidthOut   (WidthOut)
        ,.WeightWidth(WeightWidth)
        ,.BiasWidth  (BiasWidth)
        ,.InChannels (InChannels)

        ,.Weights(Weights[ch*WeightIndex +: WeightIndex])
        ,.Bias   (Biases[ch*BiasWidth +: BiasWidth])
      ) neuron_inst (
         .data_i(data_q)
        ,.data_o(data_o[ch])
      );
    end
  endgenerate

endmodule
