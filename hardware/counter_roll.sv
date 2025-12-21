
module counter_roll
#(parameter [31:0] max_val_p = 15
,parameter width_p = max_val_p <= 1 ? 'b1: $clog2(max_val_p)  
/* verilator lint_off WIDTHTRUNC */
,parameter [width_p-1:0] reset_val_p = '0
)
/* verilator lint_on WIDTHTRUNC */
(input [0:0] clk_i
,input [0:0] reset_i
,input [0:0] up_i
,input [0:0] down_i
,output [width_p-1:0] count_o);

localparam [width_p-1:0] max_val_lp = max_val_p[width_p-1:0];

logic [width_p-1:0] count_r, count_n;

assign count_o = count_r;

always_ff @(posedge clk_i) begin
// Reset immediately
	if (reset_i) begin
		count_r <= reset_val_p;
	// Otherwise iterate to next state on positive edge
	end else begin
		count_r <= count_n;
	end
end

always_comb begin
	count_n = count_r;
	// Reset has highest priority
	if (reset_i == 1'b1) begin
		count_n = reset_val_p;
	// If exclusively up and not saturated, increment
	end else if (up_i == 1'b1 && down_i == 1'b0  && count_r < max_val_lp) begin 
		count_n = count_r + 1'b1;
	// If rolling over, go to zero
	end else if (up_i == 1'b1 && down_i == 1'b0  && count_r == max_val_lp) begin 
		count_n = '0;
	// If exclusively down and not bottomed out, decrement
	end else if (up_i == 1'b0 && down_i == 1'b1 && count_r > '0) begin
		count_n = count_r - 1'b1;
	// If rolling under, go to max
	end else if (up_i == 1'b0 && down_i == 1'b1 && count_r == '0) begin
		count_n = max_val_lp;
	// Otherwise hold
	end else begin
		count_n = count_r;
	end
end

endmodule
