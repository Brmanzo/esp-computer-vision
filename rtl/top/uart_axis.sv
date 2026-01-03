`define IMAGE_W 641
`define UART_W 8
`define QUANTIZED_W 2
`define PACK_NUM 4
`timescale 1ns / 1ps
module uart_axis
	(input [0:0] clk_i // 25 MHz Clock
	,input [0:0] reset_i

	,input [0:0] rx_serial_i
	,output [0:0] tx_serial_o

	,output [5:1] led_o
	);

	// Zero pack the QUANTIZED_W from 2 to 3 bits, max pixel is 3, over 8 accumulations plus sign bit
	localparam int sobel_out_width_lp = $clog2(((((32'd1<<(`QUANTIZED_W + 1)))-1)*8)+1) + 1;

	// UART Interface Wires
	wire [0:0]                        uart_ready_w;
	wire [0:0]                        uart_valid_w;
	wire [`UART_W-1:0]                uart_data_w;

	// AXIS Adapter Wires
	wire [0:0]                        axis_ready_w;
	wire [0:0]                        axis_valid_w;
	wire [`UART_W-1:0]                axis_data_w;

	// Unpacker Wires
	wire [0:0]                        unpack_ready_w;
	wire [0:0]                        unpack_valid_w;
	wire [`QUANTIZED_W-1:0]           unpacked_data_w;

	// Sobel Wires
	wire [0:0]                        gx_ready_w, gy_ready_w;
	wire [0:0]                        gx_valid_w, gy_valid_w;
	wire [sobel_out_width_lp-1:0]     gx_data_w, gy_data_w;

	// Elastic Stage Wires
	wire [0:0]                        elastic_ready_w;
	wire [0:0]                        elastic_valid_w;
	wire [sobel_out_width_lp-1:0]     elastic_gx_w, elastic_gy_w;

	// Magnitude Wires
	wire [0:0]                        mag_ready_w;
	wire [0:0]                        mag_valid_w;
	wire [sobel_out_width_lp:0]       mag_data_w;

	// Packer Wires
	wire [0:0]                        pack_ready_w;
	wire [0:0]                        pack_valid_w;
	wire [`QUANTIZED_W*`PACK_NUM-1:0] packed_data_w;

	// Predefined Kernel weights for gx and gy gradients	
	typedef logic signed [2:0] weight_t;

	function automatic weight_t gx_w(input int unsigned i);
	unique case (i)
		0: gx_w = 3'sd1;   1: gx_w = 3'sd0;   2: gx_w = -3'sd1;
		3: gx_w = 3'sd1;   4: gx_w = 3'sd0;   5: gx_w = -3'sd1;
		6: gx_w = 3'sd1;   7: gx_w = 3'sd0;   8: gx_w = -3'sd1;
		default: gx_w = '0;
	endcase
	endfunction

	logic signed [8:0][2:0] gx_weights_w;

	genvar j;
	generate
	for (j = 0; j < 9; j++) begin : gen_gx
		assign gx_weights_w[j] = gx_w(j);  // gx_w returns weight_t (signed [2:0])
	end
	endgenerate

	function automatic weight_t gy_w(input int unsigned i);
	unique case (i)
		0: gy_w = 3'sd1;   1: gy_w = 3'sd1;   2: gy_w = 3'sd1;
		3: gy_w = 3'sd0;   4: gy_w = 3'sd0;   5: gy_w = 3'sd0;
		6: gy_w = -3'sd1;  7: gy_w = -3'sd1;  8: gy_w = -3'sd1;
		default: gy_w = '0;
	endcase
	endfunction

	logic signed [8:0][2:0] gy_weights_w;

	genvar k;
	generate
	for (k = 0; k < 9; k++) begin : gen_gy
		assign gy_weights_w[k] = gy_w(k);  // gy_w returns weight_t (signed [2:0])
	end
	endgenerate

	// For indicating FPGA operation
	assign led_o = axis_data_w[5:1];

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
	,.s_axis_tvalid(pack_valid_w)
	,.s_axis_tdata(packed_data_w)
	// UART to AXIS
	,.m_axis_tready(axis_ready_w)
	,.m_axis_tvalid(uart_valid_w)
	,.m_axis_tdata(uart_data_w)

	,.prescale(16'd27) // Fclk / (baud * 8), 25 MHz / (115200 * 8) = 27
	);

	// AXIS Adapter for UART input
	axis_adapter
	#(.S_DATA_WIDTH                   (`UART_W) // 8 bits from serial
	,.M_DATA_WIDTH                    (`UART_W) // treat as 8 bit bus to UART
	,.S_KEEP_ENABLE                   (0)
	,.M_KEEP_ENABLE                   (1)
	,.M_KEEP_WIDTH                    (1)
	,.ID_ENABLE                       (0)
	,.DEST_ENABLE                     (0)
	,.USER_ENABLE                     (0))
	axis_adapter_inst
	(.clk(clk_i)
	,.rst(reset_i)
	// UART to AXIS
	,.s_axis_tready(axis_ready_w)
	,.s_axis_tvalid(uart_valid_w)
	,.s_axis_tdata(uart_data_w)
	,.s_axis_tkeep(1'b1)
	,.s_axis_tlast(1'b0)
	// AXIS to Unpacker
	,.m_axis_tready(unpack_ready_w)
	,.m_axis_tvalid(axis_valid_w)
	,.m_axis_tdata(axis_data_w)
	);

	// Unpacker to unpack 4 2-bit values from each 8-bit UART input
	unpacker
	#(.unpacked_p(`QUANTIZED_W)
	,.num_packed_p(`PACK_NUM))
	unpacker_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// AXIS to Unpacker
	,.ready_o(unpack_ready_w)
	,.valid_i(axis_valid_w)
	,.packed_i(axis_data_w)
	// Unpacker to Sobel Filters
	,.ready_i(gx_ready_w & gy_ready_w)
	,.valid_o(unpack_valid_w)
	,.unpacked_o(unpacked_data_w)
	);

	// Sobel Filter for Gx gradient
	sobel
	#(.linewidth_px_p(`IMAGE_W)
	,.in_width_p(`QUANTIZED_W + 1)
	,.out_width_p(sobel_out_width_lp))
	sobel_gx_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Unpacker to Gx
	,.ready_o(gx_ready_w)
	,.valid_i(unpack_valid_w)
	,.data_i({1'b0, unpacked_data_w})
	// Gx to Elastic Stage
	,.ready_i(elastic_ready_w)
	,.valid_o(gx_valid_w)
	,.data_o(gx_data_w)
	,.weights_i(gx_weights_w)
	);

	// Sobel Filter for Gy gradient
	sobel
	#(.linewidth_px_p(`IMAGE_W)
	,.in_width_p(`QUANTIZED_W + 1)
	,.out_width_p(sobel_out_width_lp))
	sobel_gy_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Unpacker to Gy
	,.ready_o(gy_ready_w)
	,.valid_i(unpack_valid_w)
	,.data_i({1'b0, unpacked_data_w})
	// Gy to Elastic Stage
	,.ready_i(elastic_ready_w)
	,.valid_o(gy_valid_w)
	,.data_o(gy_data_w)
	,.weights_i(gy_weights_w)
	);

	// Elastic stage to meet timing requirements
	elastic
	#(.width_p(sobel_out_width_lp * 2) // 8 bits (4 for gx and 4 for gy)
	,.datapath_gate_p(1))
	elastic_stage_1_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Gx and Gy to Elastic Stage
	,.ready_o(elastic_ready_w)
	,.valid_i(gx_valid_w & gy_valid_w) 
	,.data_i({gx_data_w, gy_data_w})
	// Elastic Stage to Magnitude
	,.ready_i(mag_ready_w)
	,.valid_o(elastic_valid_w)
	,.data_o({elastic_gx_w, elastic_gy_w})
	);

	// Magnitude calculated from Gx and Gy
	mag
	#(.width_in_p(sobel_out_width_lp))
	mag_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Elastic Stage to Magnitude
	,.ready_o(mag_ready_w)
	,.valid_i(elastic_valid_w)
	,.gx_i(elastic_gx_w)
	,.gy_i(elastic_gy_w)
	// Magnitude to Packer
	,.ready_i(pack_ready_w)
	,.valid_o(mag_valid_w)
	,.mag_o(mag_data_w)
	);

	// Packer to pack 4 2-bit magnitude values into each 8-bit UART output
	packer
	#(.unpacked_p(`QUANTIZED_W)
	,.num_packed_p(`PACK_NUM))
	packer_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Magnitude to Packer
	,.ready_o(pack_ready_w)
	,.valid_i(mag_valid_w)
	,.unpacked_i(mag_data_w[4:3])
	// Packer to UART output
	,.ready_i(uart_ready_w)
	,.valid_o(pack_valid_w)
	,.packed_o(packed_data_w)
	);

endmodule
