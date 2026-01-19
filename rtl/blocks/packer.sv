`timescale 1ns / 1ps
module packer
	#(parameter unpacked_width_p = 2
 	,parameter  packed_num_p     = 4
	,parameter  packed_width_p   = unpacked_width_p * packed_num_p
	)
	(input  [0:0] clk_i
	,input  [0:0] reset_i

	,input  [unpacked_width_p-1:0] unpacked_i
	,input  [0:0] flush_i
	,input  [0:0] valid_i
	,output [0:0] ready_o

	,output [packed_width_p-1:0] packed_o
	,output [0:0] valid_o
	,input  [0:0] ready_i
	);

	/*  ------------------------------------ Flush Logic ------------------------------------ */
	wire  [0:0] elastic_ready_ow;
	wire  [0:0] last_w       = (counter_r == max_count_w);
	wire  [0:0] partial_w    = (counter_r != '0);

	wire  [0:0] in_fire_w    = valid_i && ready_o;
	// Block upstream data while receiving fourth input or while flushing
	assign ready_o = (last_w || flush_i) ? elastic_ready_ow : 1'b1;
	
	// Determine if flush is occurring with input
	wire  [0:0] flush_in_w      = flush_i && in_fire_w;
	wire  [0:0] flush_partial_w = flush_i && partial_w && !in_fire_w;

	// Output when either completing final pack, when flush on input, or when flushing without input
	// flush_i is decoupled from valid_i, so we can flush even if no input is valid
	// If flush_i asserted multiple cycles in a row, we want to bypass valid data to the output without packing.
	wire  [0:0] out_fire_w   = (last_w && in_fire_w) || flush_in_w || (flush_partial_w && elastic_ready_ow);
	
	/* ------------------------------------ Counter Logic ------------------------------------ */
	localparam int count_width_lp = $clog2(packed_num_p);
	logic [count_width_lp-1:0] counter_r;
	wire  [count_width_lp-1:0] max_count_w = count_width_lp'(packed_num_p - 1);

	always_ff @(posedge clk_i) begin
		if (reset_i) begin 
			counter_r <= '0;
		end else if (out_fire_w) begin
			counter_r <= '0;
		end else if (in_fire_w) begin 
			counter_r <= counter_r + 1;
		end
	end

	/* ------------------------------------ Packing Logic ------------------------------------ */
	logic [packed_width_p-1:0]         packed_r, packed_n;

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

	// Maintain offset of current shift/select within shift_reg_l
	logic [$clog2(packed_width_p)-1:0] offset_l;
	logic [packed_width_p-1:0]         shift_reg_l;

	// Apply shift and accumulate
	always_comb begin
		offset_l = counter_r * unpacked_width_p;
		shift_reg_l = {{(packed_width_p-unpacked_width_p){1'b0}}, unpacked_i};
		packed_n = packed_r | (shift_reg_l << offset_l);
	end

	/* ------------------------------------ Elastic Interface ------------------------------------ */
	wire [packed_width_p-1:0] elastic_data_iw = flush_partial_w ? packed_r : packed_n;

	elastic
	#(.width_p(packed_width_p)
	,.datapath_gate_p(1)
	,.datapath_reset_p(1)
	)
	elastic_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i(elastic_data_iw)
	,.valid_i(out_fire_w)
	,.ready_o(elastic_ready_ow)
	,.valid_o(valid_o)
	,.data_o(packed_o)
	,.ready_i(ready_i)
	);

endmodule
