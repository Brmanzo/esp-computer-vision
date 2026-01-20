`timescale 1ns / 1ps

module framer
#(parameter unpacked_width_p  = 1
 ,parameter packed_num_p = 8
 ,parameter packed_width_p = unpacked_width_p * packed_num_p
 ,parameter packet_len_elems_p = 1024 // Number of packed elements per packet
 ,parameter tail_byte_0_p = 8'h0D
 ,parameter tail_byte_1_p = 8'h0A
 )
(input  [0:0]                   clk_i
,input  [0:0]                   reset_i

,input  [0:0]                   valid_i
,output [0:0]                   ready_o
,input  [unpacked_width_p-1:0]  unpacked_i

,output [0:0]                   valid_o
,input [0:0]                    ready_i
,output [packed_width_p-1:0]    data_o
);
    // FSM States
    typedef enum logic [1:0] {forward_s, footer_0_s, footer_1_s} fsm_e;
	fsm_e state_r, state_n;

    // Packer Wires
    wire  [0:0]                pack_ready_w;
    wire  [0:0]                pack_valid_w;
    wire  [packed_width_p-1:0] packed_data_w;

    logic [0:0] last_input_seen_r;

    // Output Assignments
    assign valid_o = (state_r == forward_s) ?  pack_valid_w : 1'b1;
    // Forward data from packer until footer state, then send tail bytes
    assign data_o  = (state_r == forward_s) ?  packed_data_w :
                     (state_r == footer_0_s) ? tail_byte_0_p :
                     tail_byte_1_p;
    assign ready_o = (state_r == forward_s) && pack_ready_w && !last_input_seen_r;

    // Handshake Wires
    wire  [0:0] in_fire_w  = valid_i && ready_o;
    wire  [0:0] out_fire_w = valid_o && ready_i;

    /* ---------------------------------------- Counter Logic ---------------------------------------- */
    localparam int count_width_lp      = $clog2(packet_len_elems_p);
    wire  [count_width_lp-1:0] max_count_w = count_width_lp'(packet_len_elems_p - 1);
    logic [count_width_lp-1:0] counter_r;
    wire  [0:0] counter_max_w = (counter_r == max_count_w);

    // Saturating counter to track number of packed inputs
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            counter_r <= '0;
        // Reset counter when exiting second footer write
        end else if (state_r == footer_1_s && out_fire_w) begin
            counter_r <= '0;
        // Increment counter when accepting input in forward state
        end else if (state_r == forward_s && in_fire_w) begin
            // Saturate at max count
            if (!counter_max_w) begin
                counter_r <= counter_r + 1'b1;
            end
        end
    end

    /* ------------------------------------------- FSM Logic ------------------------------------------- */
    // Current state logic
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state_r <= forward_s;
        end else if (out_fire_w) begin
            state_r <= state_n;
        end
    end
    
    // Next state logic
    always_comb begin
        state_n = state_r;
        case (state_r)
            forward_s: begin
                if (last_input_seen_r && out_fire_w) begin
                    state_n = footer_0_s;
                end
            end
            footer_0_s: begin
                if (out_fire_w) begin
                    state_n = footer_1_s;
                end
            end
            footer_1_s: begin
                if (out_fire_w) begin
                    state_n = forward_s;
                end
            end
            default: begin
                state_n = forward_s;
            end
        endcase
    end

    // Data Logic
    logic [0:0] flush_packer_r;
    wire  [0:0] last_in_fire_w = in_fire_w && counter_max_w;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            flush_packer_r    <= 1'b0;
            last_input_seen_r <= 1'b0;
        end else begin
            // Delay flush by one cycle
            flush_packer_r <= last_in_fire_w;
            // Last footer byte sent, clear last input seen
            if (state_r == footer_1_s && out_fire_w) begin
                last_input_seen_r <= 1'b0;
            end else if (last_in_fire_w) begin
                last_input_seen_r <= 1'b1; // Last input accepted
            end
        end
    end
    /* ------------------------------------------ Packer Inst ------------------------------------------ */
    // Packer to pack 4 2-bit magnitude values into each 8-bit UART output
	packer
	#(.unpacked_width_p(unpacked_width_p)
	,.packed_num_p(packed_num_p))
	packer_inst
	(.clk_i(clk_i)
	,.reset_i(reset_i)
	// Magnitude to Packer
	,.unpacked_i(unpacked_i)
    ,.flush_i(flush_packer_r)
	,.valid_i(valid_i && !last_input_seen_r)
	,.ready_o(pack_ready_w)
	// Packer to UART output
	,.packed_o(packed_data_w)
	,.valid_o(pack_valid_w)
	,.ready_i((state_r == forward_s) ? ready_i : 1'b0)
	);

endmodule
