`define IMAGE_W 161
`define RGB_W_24 24 // Width of the expanded AXIS data from UART
`define GRAY_W_8 8
`define SOBEL_XY_W_14 14
`define MAG_W_15 15

module uart_axis
	#(parameter example_p = 0) // Does nothing, just an example. You may use it, extend it, or ignore it.
	(input [0:0] clk_i // 25 MHz Clock
	,input [0:0] reset_i

	,input [0:0] rx_serial_i
	,output [0:0] tx_serial_o

	,output [5:1] led_o // For debugging
	);

	localparam [31:0] data_width_lp = 8; // Keep this constant. Treat UART as an 8-bit bus, output.

	// In my soltion, these wires are data coming from (tx), and going
	// to (rx) the UART module. You may pick your own alternate naming
	// scheme.
	wire [data_width_lp-1:0]  m_axis_uart_tdata;
	wire [0:0]                m_axis_uart_tvalid;
	wire [0:0]                m_axis_uart_tready;

	// Expanded data, coming out of the adapter
	wire [`RGB_W_24-1:0]      m_axis_tdata;
	wire [0:0]                m_axis_tvalid;
	wire [0:0]                m_axis_tready;

	// Shrunken data, combining sobel outputs onto 8 bit bus for UART
	wire [`MAG_W_15 + 2 -1:0] s_axis_uart_tdata;
	wire [0:0]                s_axis_uart_tvalid;
	wire [0:0 ]               s_axis_uart_tready;
	
	wire [0:0]                s_axis_tready;
	// Gray data, going out to the UART
	wire [`GRAY_W_8-1:0]  gray_tdata;
	wire        gray_tvalid;

	typedef logic signed [2:0] weight_t;

	weight_t gx_weights_l [0:8] = '{
		3'sd1,  3'sd0, -3'sd1,
		3'sd2,  3'sd0, -3'sd2,
		3'sd1,  3'sd0, -3'sd1
	};

	weight_t gy_weights_l [0:8] = '{
		3'sd1,  3'sd2,  3'sd1,
		3'sd0,  3'sd0,  3'sd0,
	-3'sd1, -3'sd2, -3'sd1
	};

	wire [0:0] sobel_gx_ready_ow, sobel_gy_ready_ow;
	wire [0:0] sobel_gx_valid_ow, sobel_gy_valid_ow;

	wire signed [15:0] sobel_gx_ow, sobel_gy_ow;
	wire        [`MAG_W_15-1:0]     mag_ow;

	wire [0:0] mag_ready_ow, mag_valid_ow;

	wire [`SOBEL_XY_W_14-1:0] stage_1_gx_ow, stage_1_gy_ow;
	wire [0:0]  stage_1_valid_o, stage_1_ready_o;

	// UART head to convert UART serial data to AXIS data
	uart
	#()
	uart_inst
	(.clk(clk_i)
	,.rst(reset_i)
	
	// FPGA interface for UART 
	,.txd(tx_serial_o)
	,.rxd(rx_serial_i)
	
	// Manager (data received from wire)
	,.m_axis_tready(m_axis_uart_tready) // Input
	,.m_axis_tvalid(m_axis_uart_tvalid) 
	,.m_axis_tdata(m_axis_uart_tdata)

	// Suboordinate (data to be sent out on wire)
	,.s_axis_tready(s_axis_uart_tready) // Output
	,.s_axis_tvalid(mag_valid_ow)
	,.s_axis_tdata(mag_ow[10:3]) // This window yields the best results

	,.prescale(16'd27) // Fclk / (baud * 8), 25 MHz / (115200 * 8) = 27
	);

	// Data input as gray from software, still want axis adapter at head despite no expansion
	axis_adapter
	#(.S_DATA_WIDTH                   (data_width_lp) // 8 bits from serial
	,.M_DATA_WIDTH                    (`GRAY_W_8) // 24 bits expanded
	,.S_KEEP_ENABLE                   (0)
	,.M_KEEP_ENABLE                   (1)
	,.M_KEEP_WIDTH                    (1)
	,.ID_ENABLE                       (0)
	,.DEST_ENABLE                     (0)
	,.USER_ENABLE                     (0))
	axis_adapter_expander_inst
	(.clk(clk_i)
	,.rst(reset_i)
	// Input from UART to RGB
	,.s_axis_tready(m_axis_uart_tready)
	,.s_axis_tdata(m_axis_uart_tdata)
	,.s_axis_tkeep(1'b1)
	,.s_axis_tlast(1'b0)
	,.s_axis_tvalid(m_axis_uart_tvalid)
	// Output from Gray to UART
	,.m_axis_tready(sobel_gx_ready_ow & sobel_gy_ready_ow)
	,.m_axis_tdata(gray_tdata)
	,.m_axis_tkeep()
	,.m_axis_tvalid(gray_tvalid)
	);

	sobel
	#(.linewidth_px_p(`IMAGE_W)
	,.width_p(`GRAY_W_8))
	sobel_gx_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.valid_i(gray_tvalid)
	,.ready_o(sobel_gx_ready_ow)
	,.data_i(gray_tdata)
	,.valid_o(sobel_gx_valid_ow)
	,.ready_i(stage_1_ready_o)
	,.abs_i(1'b1)
	,.weights_i(gx_weights_l)
	,.sign_o()
	,.data_o(sobel_gx_ow)
	);

	sobel
	#(.linewidth_px_p(`IMAGE_W)
	,.width_p(`GRAY_W_8))
	sobel_gy_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.valid_i(gray_tvalid)
	,.ready_o(sobel_gy_ready_ow)
	,.data_i(gray_tdata)
	,.valid_o(sobel_gy_valid_ow)
	,.ready_i(stage_1_ready_o)
	,.abs_i(1'b1)
	,.weights_i(gy_weights_l)
	,.sign_o()
	,.data_o(sobel_gy_ow)
	);
	// Elastic stage to meet timing requirements
	elastic
	#(.width_p(`SOBEL_XY_W_14 * 2) // 16 bits for gx, 16 bits for gy
	,.datapath_gate_p(1))
	elastic_stage_1_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	,.data_i({sobel_gx_ow[13:0], sobel_gy_ow[13:0]})
	,.valid_i(sobel_gx_valid_ow & sobel_gy_valid_ow)
	,.ready_o(stage_1_ready_o)
	,.valid_o(stage_1_valid_o)
	,.data_o({stage_1_gx_ow, stage_1_gy_ow}) // When valid data, put onto shift register
	,.ready_i(mag_ready_ow)
	);

	// No shrinking necessary, just output 8 bit window from 15 bit magnitude
	mag
	#(.width_in_p(`SOBEL_XY_W_14))
	mag_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)

	,.valid_i(stage_1_valid_o)
	,.gx_i(stage_1_gx_ow)
	,.gy_i(stage_1_gy_ow)
	,.ready_o(mag_ready_ow)

	,.valid_o(mag_valid_ow)
	,.mag_o(mag_ow)
	,.ready_i(s_axis_uart_tready)
	);

	// For verifying FPGA operation, but tx will also be obervable via serial
	assign led_o = gray_tdata[5:1];

endmodule
