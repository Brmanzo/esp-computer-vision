`timescale 1ns / 1ps

module skid_buffer #(
    parameter width_p = 8,
    parameter depth_p = 16,
    parameter headroom_p = 4 // Assert RTS when only 4 slots remain
)(
    input [0:0] clk_i,
    input [0:0] reset_i,

    input  [width_p-1:0] data_i,
    input  [0:0]         valid_i,
    output [0:0]         ready_o,

    output [width_p-1:0] data_o,
    output [0:0]         valid_o,
    input  [0:0]         ready_i,

    // Flow Control
    output [0:0]         rts_o // High = STOP SENDING
);
    /* ------------------------------------ Pointer Declarations ------------------------------------ */
    logic [$clog2(depth_p)-1:0] read_ptr_r, read_ptr_n, write_ptr_r;
    logic [$clog2(depth_p)-1:0] prev_write_ptr_r = '0;
    logic [0:0] read_wrap_l, write_wrap_l;

    wire [$clog2(depth_p):0] write_n_sink_w;
    wire [0:0] read_wrap_n_sink_w;

    /* ------------------------------------ Full/Empty Logic ------------------------------------ */
    always_ff @ (posedge clk_i) begin
        prev_write_ptr_r <= write_ptr_r;
    end

    wire [0:0] empty_w = (write_ptr_r == read_ptr_r) && (write_wrap_l == read_wrap_l);
    wire [0:0] full_w  = (write_ptr_r == read_ptr_r) && (write_wrap_l != read_wrap_l);

    assign ready_o = ~full_w;
    assign valid_o = ~empty_w;

    wire [0:0] write_en_w = ready_o && valid_i;
    wire [0:0] read_en_w = ready_i && valid_o;

    // RTS Logic: Assert 'Busy' if we don't have enough headroom_p for the "skid"
    // The ESP32 might send 1-2 bytes AFTER we say stop, so we need space for them.
    assign rts_o = ({write_wrap_l, write_ptr_r} - {read_wrap_l, read_ptr_r} >= (depth_p - headroom_p));
    /* ------------------------------------ Bypass Logic ------------------------------------ */
    logic [width_p-1:0] data_bypass_w;

    // Elastic Head to latch input for bypass logic
    elastic
    #(.width_p(width_p))
    elastic_inst
    (.clk_i(clk_i)
    ,.reset_i(reset_i)
    ,.data_i(data_i)
    ,.valid_i(valid_i)
    ,.ready_o()
    ,.valid_o()
    ,.data_o(data_bypass_w)
    ,.ready_i(ready_i)
    );

    wire  [width_p-1:0] ram_data_w;
    logic [width_p-1:0] data_l;
    assign data_o = data_l;

    wire bypass_enable_w = (prev_write_ptr_r == read_ptr_r) && ~full_w;

    always_comb begin
    if (bypass_enable_w) begin
        data_l = data_bypass_w;
    end else begin
        data_l = ram_data_w;
    end
    end
    /* ------------------------------- Pointer Counter Instantiation ------------------------------- */
    // Read Counter
    counter
    #(.width_p($clog2(depth_p) + 1), // One extra bit for full/empty distinction
    .reset_val_p('0))
    read_counter_inst
    (.clk_i(clk_i)
    ,.reset_i(reset_i)
    ,.up_i(read_en_w) // Increment if ready_i and valid_o
    ,.down_i(1'b0) // Only increments then rolls over
    ,.count_o({read_wrap_l, read_ptr_r})
    ,.next_count_o({read_wrap_n_sink_w, read_ptr_n})
    );

    // Write Counter
    counter
    #(.width_p($clog2(depth_p) + 1) // One extra bit for full/empty distinction
    ,.reset_val_p('0))
    write_counter_inst
    (.clk_i(clk_i)
    ,.reset_i(reset_i)
    ,.up_i(write_en_w) // Increment if ready_o and valid_i
    ,.down_i(1'b0) // Only increments then rolls over
    ,.count_o({write_wrap_l, write_ptr_r})
    ,.next_count_o(write_n_sink_w)
    );

    /* ------------------------------------ RAM Instantiation ------------------------------------ */
    // FIFO RAM
    ram_1r1w_sync
    #(.width_p(width_p) // Width of FIFO specified by width_p
    ,.depth_p(1<<$clog2(depth_p)) // Depth of FIFO specified by depth_p
    )
    ram_1r1w_sync_inst
    (.clk_i(clk_i)
    ,.reset_i(reset_i)

    ,.wr_valid_i(write_en_w) // Disable write when bypassing
    ,.wr_data_i(data_i)
    ,.wr_addr_i(write_ptr_r)

    ,.rd_valid_i(1'b1)
    ,.rd_addr_i(read_ptr_n)
    ,.rd_data_o(ram_data_w)
    );

endmodule
