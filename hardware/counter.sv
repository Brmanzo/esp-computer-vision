module counter
	#(parameter width_p = 4,
		// Students: Using lint_off/lint_on commands to avoid lint checks,
		// will result in 0 points for the lint grade.
		/* verilator lint_off WIDTHTRUNC */
		parameter [width_p-1:0] reset_val_p = '0)
		/* verilator lint_on WIDTHTRUNC */
	 (input [0:0] clk_i
	 ,input [0:0] reset_i
	 ,input [0:0] up_i
	 ,input [0:0] down_i
	 ,output [width_p-1:0] count_o);

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
		// If exclusively up, increment
		end else if (up_i == 1'b1 && down_i == 1'b0) begin 
			count_n = count_r + 1'b1;
		// If exclusively down, decrement
		end else if (up_i == 1'b0 && down_i == 1'b1) begin
			count_n = count_r - 1'b1;
		// Otherwise hold
		end else begin
			count_n = count_r;
		end
	end

endmodule
