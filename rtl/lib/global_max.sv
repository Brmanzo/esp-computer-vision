// global_max.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module global_max #(
   parameter  int unsigned InBits      = 1
  ,parameter  int unsigned OutBits     = InBits
  ,parameter  int unsigned TermCount   = 9
  ,parameter  int unsigned InChannels  = 1
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
  wire  [0:0] last_term;
  
  /* verilator lint_off PINCONNECTEMPTY */
  counter_roll #(
     .CountBits  (CountWidth)
    ,.ResetVal   (0)
    ,.MaxVal     (TermCount - 1)
    ,.EnableDown (1'b0)
  ) term_counter_inst (
     .clk_i    (clk_i)
    ,.rst_i      (rst_i)
    ,.up_i       (in_fire)
    ,.down_i     (1'b0)
    ,.count_o    (counter_q)
    ,.next_o     ()
    ,.max_o      (last_term)
  );
  /* verilator lint_on PINCONNECTEMPTY */
  /* ------------------------ Max Value Logic ------------------------ */
  logic signed [InChannels-1:0][OutBits-1:0] max_q, max_d;

  always_ff @(posedge clk_i) begin
    if (rst_i) max_q <= '0;
    else       max_q <= max_d;
  end

  generate
    if (InBits == 1) begin : gen_binary_comparison
      always_comb begin
        max_d = max_q;
        if (in_fire) begin
          for (int ch = 0; ch < InChannels; ch++) begin
            // Initialize max with first term
            if (first_term) max_d[ch] = data_i[ch];
            // Encoding {0,1} as {-1,1} requires unsigned, as signed cast inverts comparison
            else max_d[ch] = ($unsigned(max_q[ch]) > $unsigned(data_i[ch])) ? max_q[ch] : data_i[ch];
          end
        end
      end
    end else begin : gen_signed_comparison
      always_comb begin
        max_d = max_q;
        if (in_fire) begin
          for (int ch = 0; ch < InChannels; ch++) begin
            // Initialize max with first term
            if (first_term)          max_d[ch] = data_i[ch];
            // then update if current term is greater than max
            else max_d[ch] = ($signed(max_q[ch]) > $signed(data_i[ch])) ? max_q[ch] : data_i[ch];
          end
        end
      end
    end
  endgenerate

  /* ------------------ Handshaking & Output Logic ------------------ */
  wire  [0:0] in_fire   = valid_i && ready_o;
  wire  [0:0] out_fire  = in_fire && last_term;

  logic [0:0] valid_r;
  always_ff @(posedge clk_i) begin
     if (rst_i)        valid_r <= 1'b0;
     else if (ready_o) valid_r <= out_fire;
  end

  assign valid_o = valid_r;
  assign ready_o = ~valid_r | ready_i;
  assign data_o  = max_q;


endmodule
