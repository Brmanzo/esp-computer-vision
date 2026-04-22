// global_max.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module global_max #(
   parameter  int unsigned InBits      = 1
  ,parameter  int unsigned OutBits     = InBits
  ,parameter  int unsigned TermCount   = 9
  ,parameter  int unsigned InChannels   = 1
  ,localparam int unsigned CountWidth  = (TermCount <= 1) ? 1 : $clog2(TermCount)
)  (
   input  [0:0] clk_i
  ,input  [0:0] rst_i

  ,input  [0:0] valid_i
  ,output [0:0] ready_o

  ,input  signed [InChannels-1:0][InBits-1:0] data_i

  ,output [0:0] valid_o
  ,input  [0:0] ready_i
  ,output signed [InChannels-1:0][OutBits-1:0] data_o
);

  /* ------------------------ Counter Logic ------------------------ */
  logic [CountWidth-1:0] counter_d, counter_q;

  wire  [0:0] first_term = (counter_q == '0);
  wire  [0:0] last_term = (counter_q == (CountWidth'(TermCount) - CountWidth'(1)));
  
  always_ff @(posedge clk_i) begin
    if (rst_i) counter_q <= '0;
    else       counter_q <= counter_d;
  end

  always_comb begin
    counter_d = counter_q;
    if (in_fire) begin
      // Roll over if last term,
      if (last_term) counter_d = '0;
      // Otherwise increment
      else counter_d = counter_q + CountWidth'(1);
    end
  end

  /* ------------------------ Max Value Logic ------------------------ */
  logic signed [InChannels-1:0][OutBits-1:0] max_q, max_d;

  always_ff @(posedge clk_i) begin
    if (rst_i) max_q <= '0;
    else       max_q <= max_d;
  end

  always_comb begin
    max_d = max_q;
    if (in_fire) begin
      for (int ch = 0; ch < InChannels; ch++) begin
        // Initialize max with first term
        if (first_term)          max_d[ch] = data_i[ch];
        // then update if current term is greater than max
        else if (data_i[ch] > max_q[ch]) max_d[ch] = data_i[ch];
      end
    end
  end

  /* ------------------ Elastic Handshaking Logic ------------------ */
  wire  [0:0] in_fire   = valid_i && ready_o;
  wire  [0:0] out_fire  = in_fire && last_term;

  elastic #(
     .Width        (InChannels*OutBits)
    ,.DatapathGate (1)
    ,.DatapathReset(1)
  ) elastic_inst (
     .clk_i  (clk_i)
    ,.rst_i  (rst_i)

    ,.data_i (max_d)
    ,.valid_i(out_fire)
    ,.ready_o(ready_o)

    ,.valid_o(valid_o)
    ,.data_o (data_o)
    ,.ready_i(ready_i)
  );


endmodule
