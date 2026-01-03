`timescale 1ns / 1ps
module packer
	#(parameter unpacked_p  = 2
 	,parameter num_packed_p = 4)
	(input  [0:0] clk_i
	,input  [0:0] reset_i

	,input  [unpacked_p-1:0] unpacked_i
	,input  [0:0] valid_i
	,output [0:0] ready_o

	,output [(unpacked_p*num_packed_p)-1:0] packed_o
	,output [0:0] valid_o
	,input  [0:0] ready_i
	);

	// Buffering Logic
	logic [7:0] packed_r, packed_n;

	wire  [$clog2(num_packed_p)-1:0] counter_w;

	wire  [0:0] in_fire_w = valid_i && ready_o;
	wire  [0:0] out_fire_w = (counter_w == 2'(num_packed_p - 1)) && in_fire_w;

	wire  [0:0] elastic_valid_ow, elastic_ready_ow;

	localparam [$clog2(num_packed_p)-1:0] num_packed_lp = num_packed_p[$clog2(num_packed_p)-1:0];

	// Block upstream data while receiving fourth input
	assign ready_o = (counter_w != (num_packed_lp - 1)) ? 1'b1 : elastic_ready_ow;
	// Only assert valid when all four inputs have been packed
	assign valid_o = elastic_valid_ow;

	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			packed_r <= '0;
		end else begin
			if (in_fire_w) begin
				packed_r <= packed_n;
			end
			if (out_fire_w) begin
				packed_r <= '0;
			end
		end 
	end

	counter_roll
	#(.width_p($clog2(num_packed_p))
	,.reset_val_p('0)
	)
	counter_roll_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.max_val_i(num_packed_lp-1)
	,.up_i(in_fire_w)
	,.down_i('0)
	,.count_o(counter_w)
	);

	elastic
	#(.width_p(8)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(packed_n)
	,.valid_i(out_fire_w)
	,.ready_o(elastic_ready_ow)
	,.valid_o(elastic_valid_ow)
	,.data_o(packed_o)
	,.ready_i(ready_i)
	);

	// Packing Logic
	always_comb begin
		packed_n = packed_r;
		case (counter_w)
			2'd0: packed_n    = packed_r | {6'b0, unpacked_i      };
			2'd1: packed_n    = packed_r | {4'b0, unpacked_i, 2'b0};
			2'd2: packed_n    = packed_r | {2'b0, unpacked_i, 4'b0};
			2'd3: packed_n    = packed_r | {      unpacked_i, 6'b0};
			default: packed_n = packed_r;
		endcase
	end

endmodule
