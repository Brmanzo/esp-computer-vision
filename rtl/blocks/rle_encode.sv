module rle_encode
    #(parameter data_width_p = 2
    ,parameter bus_width_p   = 8
    ,parameter count_width_p = bus_width_p - data_width_p
    )
	(input  [0:0] clk_i
	,input  [0:0] reset_i

	,input  [data_width_p-1:0] data_i
	,input  [0:0] valid_i
	,output [0:0] ready_o

	,output [data_width_p-1:0]  rle_value_o
	,output [count_width_p-1:0] rle_count_o
	,output [0:0] valid_o
	,input  [0:0] ready_i
	);

	logic [data_width_p-1:0]  cur_value_r, cur_value_n;
	logic [count_width_p-1:0] cur_count_r, cur_count_n;

	wire  [0:0] in_fire_w  = valid_i && ready_o;
	wire  [0:0] out_fire_w = valid_o && ready_i;

	wire  [0:0] elastic_valid_ow, elastic_ready_ow;

	// Block upstream data while receiving fourth input
	assign ready_o = (counter_w != (num_packed_lp - 1)) ? 1'b1 : elastic_ready_ow;
	// Only assert valid when all four inputs have been packed
	assign valid_o = elastic_valid_ow;

	always_ff @(posedge clk_i) begin
		if (reset_i) begin
			cur_count_r <= '0;
			cur_value_r <= '0;
		end else if (in_fire_w) begin
			cur_count_r <= cur_count_n;
			cur_value_r <= cur_value_n;
		end
	end

	always_comb begin
		cur_value_n = cur_value_r;
		if (in_fire_w) begin
			if (data_i == cur_value_r) begin
				cur_count_n = cur_count_r + 1'b1;
				// Count maxed out, emit current and start new
				if (cur_count_r == {count_width_p{1'b1}}) begin
					
				end
			end else begin
				cur_value_n = data_i;
				cur_count_n = '1;
			end
		end
	end

	elastic
	#(.width_p(bus_width_p)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i({cur_count_r, cur_value_r})
	,.valid_i(out_fire_w)
	,.ready_o(elastic_ready_ow)
	,.valid_o(elastic_valid_ow)
	,.data_o(packed_o)
	,.ready_i(ready_i)
	);

endmodule
