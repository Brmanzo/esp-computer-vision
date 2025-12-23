module unpacker
	#(parameter unpacked_p  = 2
	,parameter num_packed_p = 4)
	(input  [0:0] clk_i
	,input  [0:0] reset_i

	,input  [7:0] packed_i
	,input  [0:0] valid_i
	,output [0:0] ready_o

	,output [1:0] unpacked_o
	,output [0:0] valid_o
	,input  [0:0] ready_i
	);

	// Buffering Logic
	logic [7:0] packed_buf_l;
	logic [7:0] unpacked_ol;
	
	logic [0:0] unpacking_l;

	wire  [0:0] elastic_ready_ow;

	wire  [$clog2(num_packed_p)-1:0] counter_w;
	wire  [0:0] uart_fire_w = valid_i && ready_o;
	wire  [0:0] unpack_fire_w = unpacking_l && elastic_ready_ow;
	wire  [0:0] unpack_done_w = (counter_w == 2'(num_packed_p-1)) && unpack_fire_w;

	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			packed_buf_l <= 0;
			unpacking_l <= 1'b0;
		end else begin
			if (uart_fire_w) begin
				packed_buf_l <= packed_i;
				unpacking_l <= 1'b1;
			end else if (unpack_done_w && !uart_fire_w) begin
				unpacking_l <= 1'b0;
			end
		end 
	end


	assign ready_o = (~unpacking_l) || unpack_done_w;

	counter_roll
	#(.max_val_p(3)
	,.reset_val_p('0)
	)
	counter_roll_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.up_i(unpack_fire_w)
	,.down_i('0)
	,.count_o(counter_w)
	);

	elastic
	#(.width_p(2)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(unpacked_ol[unpacked_p-1:0])
	,.valid_i(unpacking_l)
	,.ready_o(elastic_ready_ow)
	,.valid_o(valid_o)
	,.data_o(unpacked_o)
	,.ready_i(ready_i)
	);

	// Unpacking Logic
	always_comb begin
		case (counter_w)
			2'd0: unpacked_ol =  packed_buf_l       & 8'h03;
			2'd1: unpacked_ol = (packed_buf_l >> 2) & 8'h03;
			2'd2: unpacked_ol = (packed_buf_l >> 4) & 8'h03;
			2'd3: unpacked_ol = (packed_buf_l >> 6) & 8'h03;
			default: unpacked_ol = '0;
		endcase
	end

endmodule
