`timescale 1ns / 1ps
module elastic
	#(parameter [31:0] width_p = 8
	/* verilator lint_off WIDTHTRUNC */
		,parameter [0:0] datapath_gate_p = 0
		,parameter [0:0] datapath_reset_p = 0
		)
	(input [0:0] clk_i
	,input [0:0] reset_i

	,input [width_p - 1:0] data_i
	,input [0:0] valid_i
	,output [0:0] ready_o 

	,output [0:0] valid_o 
	,output [width_p - 1:0] data_o 
	,input [0:0] ready_i
	);

	// State Logic
	typedef enum logic [0:0] {idle_s, ihvd_s} fsm_e;
	fsm_e state_r, state_n;

	// DFF
	logic [width_p - 1:0] data_r, data_n;

	assign ready_o = (state_r == idle_s) ||
						  (state_r == ihvd_s && ready_i);

	assign valid_o = (state_r == ihvd_s);
	assign data_o  = data_r;

	// Current state logic
	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			// Always initialize idle state on reset
			state_r <= idle_s;
			// Only reset data if parameter is set
			if (datapath_reset_p) begin
				data_r <= '0;
			end
		end else begin
			state_r <= state_n;
			data_r <= data_n;
		end
	end

	// Next state logic
	always_comb begin
		state_n = state_r;

		case (state_r)
			idle_s: begin
				if (valid_i) begin
					state_n = ihvd_s;
				end
			end
			ihvd_s: begin
				if ((~valid_i && ready_i)) begin
					state_n = idle_s;
				end
			end
		endcase
	end

	// Data logic
	always_comb begin
		data_n = data_r;
		if (datapath_gate_p) begin
				if (ready_o && valid_i) begin
					data_n = data_i;
				end 
		end else begin
			if (ready_o) begin
				data_n = data_i;
			end 
		end
	end

endmodule
