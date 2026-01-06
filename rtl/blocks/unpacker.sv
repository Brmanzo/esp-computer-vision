`timescale 1ns / 1ps
module unpacker
	#(parameter unpacked_width_p  = 2
	 ,parameter packed_num_p = 4
	 ,parameter packed_width_p = unpacked_width_p * packed_num_p)
	(input  [0:0]                  clk_i
	,input  [0:0]                  reset_i

	,input  [packed_width_p-1:0]   packed_i
	,input  [0:0]                  valid_i
	,output [0:0]                  ready_o

	,output [unpacked_width_p-1:0] unpacked_o
	,output [0:0]                  valid_o
	,input  [0:0]                  ready_i
	);
	localparam int CW = $clog2(packed_num_p);
	wire [CW-1:0] max_count_w = CW'(packed_num_p - 1);

	// Buffering Logic
	logic [packed_width_p-1:0] packed_buf_r;
	logic [packed_width_p-1:0] unpacked_ol;
	
	logic [0:0] unpacking_r;
	logic [$clog2(packed_width_p)-1:0] offset_l;

	wire  [0:0] elastic_ready_ow;

	wire  [$clog2(packed_num_p)-1:0] counter_w;

	wire  [0:0] in_fire_w = valid_i && ready_o;
	wire  [0:0] out_fire_w = unpacking_r && elastic_ready_ow;
	wire  [0:0] last_w = (counter_w == max_count_w);
	wire  [0:0] done_w = last_w && out_fire_w;


	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			packed_buf_r <= 0;
			unpacking_r <= 1'b0;
		end else begin
			if (in_fire_w) begin
				packed_buf_r <= packed_i;
				unpacking_r <= 1'b1;
			end else if (done_w && !in_fire_w) begin
				unpacking_r <= 1'b0;
			end
		end 
	end


	assign ready_o = (~unpacking_r) || done_w;

	counter_roll
	#(.width_p($clog2(packed_num_p))
	,.reset_val_p('0)
	)
	counter_roll_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.max_val_i(max_count_w)
	,.up_i(out_fire_w)
	,.down_i('0)
	,.count_o(counter_w)
	);

	elastic
	#(.width_p(unpacked_width_p)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(unpacked_ol[unpacked_width_p-1:0])
	,.valid_i(unpacking_r)
	,.ready_o(elastic_ready_ow)
	,.valid_o(valid_o)
	,.data_o(unpacked_o)
	,.ready_i(ready_i)
	);

	// Mask to select the unpacked data from buffer
	wire [packed_width_p-1:0] mask_l = {{(packed_width_p - unpacked_width_p){1'b0}},
										{unpacked_width_p{1'b1}}}; // Replicating proper bit width
	
	// Unpacking Logic
	always_comb begin
		offset_l    = counter_w * unpacked_width_p;
		unpacked_ol = (packed_buf_r >> offset_l) & mask_l;
	end

endmodule
