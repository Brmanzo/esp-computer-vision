`define IMAGE_W 320 // Change with software capture width
`define IMAGE_H 240 // Change with software capture height
`define UART_W 8
`define WC_SUM 25

`define QUANTIZED_W 1
`define PACK_NUM 8
`define KERNEL_W 3
`define WEIGHT_W 2
`timescale 1ns / 1ps
module uart_axis
	(input [0:0] clk_i // 25 MHz Clock
	,input [0:0] reset_i
	,input [3:1] button_i

	,input [0:0] rx_serial_i
	,output [0:0] tx_serial_o
	,output [0:0] uart_rts_o

	,output [5:1] led_o
	);

	// Zero pack the QUANTIZED_W from 1 to 2 bits, max pixel value is 1, worst case sum is 9 per convolution
	localparam int conv2d_out_width_lp = $clog2(((((32'd1<<(`QUANTIZED_W + 1)))-1)*`WC_SUM)+1) + 1;

	// UART Interface Wires
	wire [0:0]                        uart_ready_w;
	wire [0:0]                        uart_valid_w;
	wire [`UART_W-1:0]                uart_data_w;

	// Skid Buffer Wires
	wire [0:0]                        skid_ready_w;
	wire [0:0]                        skid_valid_w;
	wire [`UART_W-1:0]                skid_data_w;

	// Unpacker Wires
	wire [0:0]                        unpack_ready_w;
	wire [0:0]                        unpack_valid_w;
	wire [`QUANTIZED_W-1:0]           unpacked_data_w;

	// conv2d Wires
	wire [0:0]                        gx_ready_w, gy_ready_w;
	wire [0:0]                        gx_valid_w, gy_valid_w;
	wire [conv2d_out_width_lp-1:0]     gx_data_w, gy_data_w;

	// Magnitude Wires
	wire [0:0]                        mag_ready_w;
	wire [0:0]                        mag_valid_w;
	wire [conv2d_out_width_lp:0]       mag_data_w;

	// Mux Logic
	logic [0:0]                       output_mux_l;
	logic [0:0]                       mux_valid_l;

	// Framer Wires
	wire [0:0]   				      framer_ready_w;
	wire [0:0]   				      framer_valid_w;
	wire [`UART_W-1:0]   			  framer_data_w;

	localparam int out_width_lp  = `IMAGE_W - (`KERNEL_W - 1);
	localparam int out_height_lp = `IMAGE_H - (`KERNEL_W - 1);

	// Predefined Kernel weights for gx and gy gradients	
	typedef logic signed [`WEIGHT_W-1:0] weight_t;

	function automatic weight_t gx_w(input int unsigned i);
		unique case (i)
			0: gx_w = 2'sd1;  1: gx_w = 2'sd0;  2: gx_w = -2'sd1;
			3: gx_w = 2'sd1;  4: gx_w = 2'sd0;  5: gx_w = -2'sd1;
			6: gx_w = 2'sd1;  7: gx_w = 2'sd0;  8: gx_w = -2'sd1;
			default: gx_w = '0;
		endcase
	endfunction

	logic signed [`KERNEL_W*`KERNEL_W-1:0][`WEIGHT_W-1:0] gx_weights_w;

	genvar j;
	generate
		for (j = 0; j < `KERNEL_W*`KERNEL_W; j++) begin : gen_gx
			assign gx_weights_w[j] = gx_w(j);  // gx_w returns weight_t (signed [2:0])
		end
	endgenerate

	function automatic weight_t gy_w(input int unsigned i);
		unique case (i)
			0: gy_w =  2'sd1;  1: gy_w =  2'sd1;  2: gy_w =  2'sd1; 
			3: gy_w =  2'sd0;  4: gy_w =  2'sd0;  5: gy_w =  2'sd0;
			6: gy_w = -2'sd1;  7: gy_w = -2'sd1;  8: gy_w = -2'sd1;
			default: gy_w = '0;
		endcase
	endfunction

	logic signed [`KERNEL_W*`KERNEL_W-1:0][`WEIGHT_W-1:0] gy_weights_w;

	genvar k;
	generate
		for (k = 0; k < `KERNEL_W*`KERNEL_W; k++) begin : gen_gy
			assign gy_weights_w[k] = gy_w(k);  // gy_w returns weight_t (signed [2:0])
		end
	endgenerate

	// For indicating FPGA operation
	// assign led_o = axis_data_w[5:1];

	// UART head to convert UART serial data to AXIS data
	/* verilator lint_off PINMISSING */
	uart
	#()
	uart_inst
	(.clk(clk_i)
	,.rst(reset_i)
	
	// FPGA interface for UART
	,.txd(tx_serial_o) // ESP_TX_o pin 2
	,.rxd(rx_serial_i) // ESP_RX_i pin 4
	// Packer to UART
	,.s_axis_tready(uart_ready_w)
	,.s_axis_tvalid(framer_valid_w)
	,.s_axis_tdata(framer_data_w)
	// UART to AXIS
	,.m_axis_tready(skid_ready_w)
	,.m_axis_tvalid(uart_valid_w)
	,.m_axis_tdata(uart_data_w)

	,.prescale(16'd10) // Fclk / (baud * 8), 25 MHz / (312,500 * 8) = 20 //10
	);

	skid_buffer
	#(.width_p(`UART_W)
	 ,.depth_p(16)
	 ,.headroom_p(6))
	skid_inst (
	.clk_i(clk_i)
	,.reset_i(reset_i)
	// UART to Skid Buffer
	,.data_i(uart_data_w)
	,.valid_i(uart_valid_w)
	,.ready_o(skid_ready_w)
	// Skid Buffer to Unpacker
	,.data_o(skid_data_w)
	,.valid_o(skid_valid_w)
	,.ready_i(unpack_ready_w)
	,.rts_o(uart_rts_o) 
    );

	// Unpacker to unpack 4 2-bit values from each 8-bit UART input
	unpacker
	#(.unpacked_width_p(`QUANTIZED_W)
	,.packed_num_p(`PACK_NUM))
	unpacker_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Skid Buffer to Unpacker
	,.ready_o(unpack_ready_w)
	,.valid_i(skid_valid_w)
	,.packed_i(skid_data_w)
	// Unpacker to conv2d Filters
	,.ready_i(gx_ready_w & gy_ready_w)
	,.valid_o(unpack_valid_w)
	,.unpacked_o(unpacked_data_w)
	);

	// conv2d Filter for Gx gradient
	conv2d
	#(.linewidth_px_p(`IMAGE_W)
	,.linecount_px_p(`IMAGE_H)
	,.in_width_p(`QUANTIZED_W + 1) // Zero pad inputs
	,.out_width_p(conv2d_out_width_lp)
	,.kernel_width_p(`KERNEL_W)
	,.weight_width_p(`WEIGHT_W))
	conv2d_gx_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Unpacker to Gx
	,.ready_o(gx_ready_w)
	,.valid_i(unpack_valid_w)
	,.data_i({1'b0, unpacked_data_w}) // "Right shift" by 3 to divide by 8 and average the output
	// Gx to Elastic Stage
	,.ready_i(mag_ready_w)
	,.valid_o(gx_valid_w)
	,.data_o(gx_data_w)
	,.weights_i(gx_weights_w)
	);

	// conv2d Filter for Gy gradient
	conv2d
	#(.linewidth_px_p(`IMAGE_W)
	,.linecount_px_p(`IMAGE_H)
	,.in_width_p(`QUANTIZED_W + 1)
	,.out_width_p(conv2d_out_width_lp)
	,.kernel_width_p(`KERNEL_W)
	,.weight_width_p(`WEIGHT_W))
	conv2d_gy_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Unpacker to Gy
	,.ready_o(gy_ready_w)
	,.valid_i(unpack_valid_w)
	,.data_i({1'b0, unpacked_data_w})
	// Gy to Elastic Stage
	,.ready_i(mag_ready_w)
	,.valid_o(gy_valid_w)
	,.data_o(gy_data_w)
	,.weights_i(gy_weights_w)
	);

	// Magnitude calculated from Gx and Gy
	mag
	#(.width_in_p(conv2d_out_width_lp))
	mag_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Elastic Stage to Magnitude
	,.ready_o(mag_ready_w)
	,.valid_i(gx_valid_w & gy_valid_w)
	,.gx_i(gx_data_w)
	,.gy_i(gy_data_w)
	// Magnitude to Packer
	,.ready_i(framer_ready_w)
	,.valid_o(mag_valid_w)
	,.mag_o(mag_data_w)
	);

	always_comb begin
		case (button_i)
			3'b001: begin output_mux_l  = unpacked_data_w; mux_valid_l = unpack_valid_w; end
			3'b010: begin output_mux_l  = gx_data_w[2];    mux_valid_l = mag_valid_w; end
			3'b100: begin output_mux_l  = gy_data_w[2];    mux_valid_l = mag_valid_w; end
			default: begin output_mux_l = mag_data_w[2];   mux_valid_l = mag_valid_w; end
		endcase
	end
	// Packs 8 pixels onto a single byte, adds footer at end of frame and sends to UART
	framer
	#(.unpacked_width_p(`QUANTIZED_W)
	,.packed_num_p(`PACK_NUM)
	,.packet_len_elems_p(out_height_lp * out_width_lp)
	,.tail_byte_0_p(8'h0D)
	,.tail_byte_1_p(8'h0A))
	framer_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Mux to Framer
	,.unpacked_i(output_mux_l)
	,.valid_i(mux_valid_l)
	,.ready_o(framer_ready_w)
	// Framer to Packer
	,.data_o(framer_data_w)
	,.valid_o(framer_valid_w)
	,.ready_i(uart_ready_w)
	);

endmodule
