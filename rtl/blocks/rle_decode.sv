module rle_decode
    #(parameter data_width_p = 2
    ,parameter bus_width_p   = 8
    ,parameter count_width_p = bus_width_p - data_width_p
    )
    (input  [0:0] clk_i
    ,input  [0:0] reset_i

    ,input  [data_width_p-1:0]  rle_value_i
    ,input  [count_width_p-1:0] rle_count_i
    ,input  [0:0]               valid_i
    ,output [0:0]               ready_o

    ,output [data_width_p-1:0]  data_o
    ,output [0:0]               valid_o
    ,input  [0:0]               ready_i
    );

    logic [data_width_p-1:0]  rle_value_r;
    logic [count_width_p-1:0] rle_max_r;

    wire [count_width_p-1:0] rle_count_w;

    wire [0:0] elastic_ready_ow;

    logic [0:0] decoding_r; // Busy


    wire [0:0] rle_fire_w = valid_i && ready_o;
    wire [0:0] emit_fire_w = decoding_r && elastic_ready_ow;

    wire [0:0] emit_done_w = (rle_count_w == (rle_max_r - 1)) && emit_fire_w;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            rle_value_r <= '0;
            rle_max_r   <= '0;
            decoding_r  <= 1'b0;
        end else begin
            if (rle_fire_w) begin
                rle_value_r <= rle_value_i;
                rle_max_r   <= rle_count_i;
                decoding_r  <= (rle_count_i != '0);
            end 
            else if (emit_done_w && !rle_fire_w) begin
                decoding_r <= 1'b0;
            end
        end
    end

    assign ready_o = (~decoding_r) || emit_done_w;

    elastic
    #(.width_p(data_width_p)
    ,.datapath_gate_p(1)
    ,.datapath_reset_p(1)
    )
    elastic_inst
    (.clk_i(clk_i)
    ,.reset_i(reset_i)

    ,.data_i(rle_value_r)
    ,.valid_i(decoding_r)
    ,.ready_o(elastic_ready_ow)

    ,.valid_o(valid_o)
    ,.data_o(data_o)
    ,.ready_i(ready_i)
    );

    counter_roll
	#(.width_p(count_width_p)
	,.reset_val_p('0)
	)
	read_counter_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.max_val_i(rle_max_r - 1)
	,.up_i(emit_fire_w)
	,.down_i('0)
	,.count_o(rle_count_w)
	);

endmodule
