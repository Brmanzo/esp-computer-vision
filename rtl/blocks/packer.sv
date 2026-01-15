`timescale 1ns / 1ps
module packer
	#(parameter unpacked_width_p = 2
 	,parameter  packed_num_p     = 4
	,parameter  packed_width_p   = unpacked_width_p * packed_num_p
	)
	(input  [0:0] clk_i
	,input  [0:0] reset_i

	,input  [unpacked_width_p-1:0] unpacked_i
	,input  [0:0] valid_i
	,output [0:0] ready_o

	,output [packed_width_p-1:0] packed_o
	,output [0:0] valid_o
	,input  [0:0] ready_i
	);

	// Counter logic
	localparam int count_width_lp = $clog2(packed_num_p);
	wire  [count_width_lp-1:0] counter_w;
	wire  [count_width_lp-1:0] max_count_w = count_width_lp'(packed_num_p - 1);

	// Packing Logic
	logic [packed_width_p-1:0]         packed_r, packed_n;
	logic [packed_width_p-1:0]         shift_reg_l;
	// Maintain offset of current shift/select within shift_reg_l
	logic [$clog2(packed_width_p)-1:0] offset_l;

	// Handshaking logic
	wire  [0:0] elastic_ready_ow;
	wire  [0:0] last_w = (counter_w == max_count_w);
	wire  [0:0] in_fire_w = valid_i && ready_o;
	wire  [0:0] out_fire_w = last_w && in_fire_w;

	always_ff @(posedge clk_i) begin
        if (reset_i) begin
            packed_r <= '0;
        end else if (out_fire_w) begin
            // We just sent the full word to elastic; clear buffer for next frame
            packed_r <= '0;
        end else if (in_fire_w) begin
            // We accepted a partial chunk; accumulate it
            packed_r <= packed_n;
        end
    end

	// Block upstream data while receiving fourth input
	assign ready_o = last_w ? elastic_ready_ow : 1'b1;
	// Only assert valid when all four inputs have been packed

	counter_roll
	#(.width_p($clog2(packed_num_p))
	,.reset_val_p('0)
	)
	counter_roll_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.max_val_i(max_count_w)
	,.up_i(in_fire_w)
	,.down_i('0)
	,.count_o(counter_w)
	);

	elastic
	#(.width_p(packed_width_p)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(packed_n)
	,.valid_i(out_fire_w)
	,.ready_o(elastic_ready_ow)
	,.valid_o(valid_o)
	,.data_o(packed_o)
	,.ready_i(ready_i)
	);

	always_comb begin
		offset_l = counter_w * unpacked_width_p;
		shift_reg_l = {{(packed_width_p-unpacked_width_p){1'b0}}, unpacked_i};
		packed_n = packed_r | (shift_reg_l << offset_l);
	end

endmodule
