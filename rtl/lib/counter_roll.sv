// counter_roll.sv
// Bradley Manzo, 2026

`timescale 1ns / 1ps
module counter_roll #(
  parameter int unsigned  CountBits  = 8
 ,parameter int unsigned  MaxVal     = '1
 ,parameter int unsigned  ResetVal   = '0
 ,parameter bit           EnableDown = 1'b1
)  (
  input [0:0] clk_i
 ,input [0:0] rst_i

 ,input [0:0] up_i
 ,input [0:0] down_i

 ,output [CountBits-1:0] count_o
 ,output [CountBits-1:0] next_o
 ,output [0:0]           max_o
);

 logic [CountBits-1:0] count_q, count_d;
 assign count_o      = count_q;
 assign next_o       = count_d;
 assign max_o        = (count_q == CountBits'(MaxVal));

 always_ff @(posedge clk_i) begin
   count_q <= count_d;
 end

 always_comb begin
   if (rst_i) begin
     count_d = CountBits'(ResetVal);
   end else begin
     count_d = count_q;
     
     if (EnableDown) begin
       // Dual direction logic
       case ({up_i, down_i})
         2'b10: begin // Increment
           /* verilator lint_off UNSIGNED */
           if (count_q < CountBits'(MaxVal)) count_d = count_q + 1'b1;
           else                              count_d = '0;
           /* verilator lint_on UNSIGNED */
         end
         2'b01: begin // Decrement
           if (count_q > '0) count_d = count_q - 1'b1;
           else              count_d = CountBits'(MaxVal);
         end
         default: count_d = count_q; // Hold on 00 or 11
       endcase
     end else begin
       // Up-only logic (saves LUTs)
       if (up_i) begin
         /* verilator lint_off UNSIGNED */
         if (count_q < CountBits'(MaxVal)) count_d = count_q + 1'b1;
         else                              count_d = '0;
         /* verilator lint_on UNSIGNED */
       end
     end
   end
 end

endmodule

