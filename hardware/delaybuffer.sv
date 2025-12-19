module delaybuffer
	#(parameter [31:0] width_p = 8
		,parameter [31:0] delay_p = 8
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
	/* verilator lint_off WIDTHTRUNC */
	logic [$clog2(delay_p):0] read_ptr_r;

	wire [0:0] put_data_w;
	assign put_data_w = (ready_o && valid_i);

	logic [width_p-1:0] elastic_data_d, elastic_data_q;

	elastic
	#(.width_p(width_p)
	,.datapath_gate_p(1)
	)
	elastic_head_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(data_i)
	,.valid_i(valid_i)
	,.ready_o(ready_o)
	,.valid_o(valid_o)
	,.data_o(elastic_data_d) // When valid data, put onto shift register
	,.ready_i(ready_i)
	);

	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			elastic_data_q <= '0;
		end else if (put_data_w) begin
			elastic_data_q <= elastic_data_d;
		end
	end

	counter_roll
	#(.max_val_p(delay_p <= 'b1 ? 'b1 : delay_p - 1)
	,.reset_val_p('0)
	)
	read_counter_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.up_i(put_data_w)
	,.down_i('0)
	,.count_o(read_ptr_r)
	);

	logic [width_p-1:0] ram_data_w;

	ram_1r1w_sync
	#(.width_p(width_p)
	,.depth_p(delay_p)
	)
	ram_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.wr_valid_i(put_data_w)
	,.wr_data_i(data_i)
	,.wr_addr_i(read_ptr_r)
	,.rd_valid_i(put_data_w)
	,.rd_addr_i(read_ptr_r)
	,.rd_data_o(ram_data_w)
	);

	assign data_o = (delay_p == 1) ? elastic_data_q : ram_data_w;
endmodule
