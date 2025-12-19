module mag
	#(
	 // This is here to help, but we won't change it.
	 parameter width_in_p = 14
	 )
	(input [0:0] clk_i
	,input [0:0] reset_i

	,input [0:0] valid_i
	,input signed [width_in_p - 1:0] gx_i
	,input signed [width_in_p - 1:0] gy_i
	,output signed [0:0] ready_o

	,output [0:0] valid_o
	,output [width_in_p:0] mag_o
	,input [0:0] ready_i
	);
	// Your code here

	logic [width_in_p - 1:0] gx_ol, gy_ol;
	logic [0:0] ready_ol;
	assign ready_o = ready_ol;

	logic [width_in_p:0] mag_n; // One extra bit to avoid overflow on shift and add
	assign mag_o = mag_n;

	elastic
	#(.width_p((width_in_p)*2)
	 ,.datapath_gate_p(1)
	 ,.datapath_reset_p(1)
	 )
	elastic_inst
	(.clk_i(clk_i)
	 ,.reset_i(reset_i)
	 ,.data_i({gx_i, gy_i})
	 ,.valid_i(valid_i)
	 ,.ready_o(ready_ol)
	 ,.valid_o(valid_o)
	 ,.data_o({gx_ol, gy_ol})
	 ,.ready_i(ready_i)
	);

	// Comparison to preserve extra bit being shifted
	always_comb begin
		if ({1'b0, gy_ol} >= {gx_ol, 1'b0}) begin
			mag_n = {1'b0, gy_ol};
		end else if ({1'b0, gx_ol} >= {gy_ol, 1'b0}) begin
			mag_n = {1'b0, gx_ol};
		// If both multiply x by 1.5 (approx of sqrt)
		end else begin
			mag_n = {1'b0,gy_ol} + {1'b0, (gx_ol >> 1)};
		end
	end

endmodule
