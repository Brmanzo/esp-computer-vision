`timescale 1ns / 1ps

module skid_buffer #(
    parameter DATA_WIDTH = 8,
    parameter DEPTH = 16,
    parameter HEADROOM = 4 // Assert RTS when only 4 slots remain
)(
    input  wire clk,
    input  wire rst,

    // Input (from UART RX)
    input  wire [DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    // Output (to Downstream)
    output wire [DATA_WIDTH-1:0] m_axis_tdata,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready,

    // Flow Control
    output wire                  rts // High = STOP SENDING
);

    // Internal storage
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] write_ptr = 0;
    reg [$clog2(DEPTH):0] read_ptr = 0;
    
    wire [$clog2(DEPTH):0] count = write_ptr - read_ptr;
    
    // RTS Logic: Assert 'Busy' if we don't have enough headroom for the "skid"
    // The ESP32 might send 1-2 bytes AFTER we say stop, so we need space for them.
    assign rts = (count >= (DEPTH - HEADROOM));

    // Standard FIFO Logic
    wire empty = (write_ptr == read_ptr);
    wire full  = (count == DEPTH);

    assign s_axis_tready = !full;
    assign m_axis_tvalid = !empty;
    assign m_axis_tdata  = mem[read_ptr[($clog2(DEPTH)-1):0]];

    always @(posedge clk) begin
        if (rst) begin
            write_ptr <= 0;
            read_ptr <= 0;
        end else begin
            // Write
            if (s_axis_tvalid && s_axis_tready) begin
                mem[write_ptr[($clog2(DEPTH)-1):0]] <= s_axis_tdata;
                write_ptr <= write_ptr + 1;
            end
            // Read
            if (m_axis_tvalid && m_axis_tready) begin
                read_ptr <= read_ptr + 1;
            end
        end
    end
endmodule
