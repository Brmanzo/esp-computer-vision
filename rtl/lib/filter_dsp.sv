// filter_dsp.sv
// Bradley Manzo, 2026
// Sequential filter implementation utilizing iCE40 SB_MAC16 DSPs

`timescale 1ns/1ps
module filter_dsp #(
    parameter   int unsigned InBits      = 1
   ,parameter   int unsigned OutBits     = 1
   ,parameter   int unsigned KernelWidth = 3
   ,parameter   int unsigned WeightBits  = 2
   ,parameter   int unsigned BiasBits    = 8
   ,parameter   int unsigned AccBits     = 32
   ,parameter   int unsigned InChannels  = 4
   ,parameter   logic signed [BiasBits-1:0] Bias = '0
   ,localparam  int unsigned KernelArea  = KernelWidth * KernelWidth
   ,localparam  int unsigned TotalTerms  = InChannels * KernelArea
   ,localparam  int unsigned TermBits    = (TotalTerms > 1) ? $clog2(TotalTerms) : 1
) (
    input [0:0] clk_i
   ,input [0:0] rst_i

   ,input [0:0] valid_i
   ,output [0:0] ready_o

   ,input logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_i
   ,input logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_i

   ,output [0:0] valid_o
   ,input  [0:0] ready_i
   ,output logic signed [OutBits-1:0] data_o
);

    /* ------------------------ Sequential Logic ------------------------ */
    logic [TermBits-1:0] term_counter;
    logic [0:0] busy_q, valid_r;
    
    // Capture registers for the parallel window and weights
    logic signed [InChannels-1:0][KernelArea-1:0][InBits-1:0]     windows_q;
    // Weights are usually parameters, but we capture them to simplify the mux
    logic signed [InChannels-1:0][KernelArea-1:0][WeightBits-1:0] weights_q;

    wire first_term = (term_counter == '0);
    wire last_term  = (term_counter == TermBits'(TotalTerms - 1));

    assign ready_o = ~busy_q & ~valid_r;
    assign valid_o =  valid_r;

    wire in_fire  = valid_i && ready_o;
    wire out_fire = valid_o && ready_i;

    always_ff @(posedge clk_i) begin
        if (rst_i) begin
            term_counter <= '0;
            busy_q       <= 1'b0;
            valid_r      <= 1'b0;
        end else begin
            if (in_fire) begin
                windows_q    <= windows_i;
                weights_q    <= weights_i;
                busy_q       <= 1'b1;
                term_counter <= '0;
            end else if (busy_q) begin
                if (last_term) begin
                    busy_q  <= 1'b0;
                    valid_r <= 1'b1;
                end else begin
                    term_counter <= term_counter + 1;
                end
            end

            if (out_fire) begin
                valid_r <= 1'b0;
            end
        end
    end

    /* ------------------------- DSP Mapping ------------------------- */
    // Flatten the captured data for the multiplexer
    wire signed [TotalTerms-1:0][InBits-1:0]     flat_windows = windows_q;
    wire signed [TotalTerms-1:0][WeightBits-1:0] flat_weights = weights_q;

    // Force DSP inference by widening the operands to 16 bits.
    wire signed [15:0] data_w   = 16'($signed(flat_windows[term_counter]));
    wire signed [15:0] weight_w = 16'($signed(flat_weights[term_counter]));
    wire signed [15:0] data_in_w   = 16'($signed(windows_i[0][0]));
    wire signed [15:0] weight_in_w = 16'($signed(weights_i[0][0]));

    logic signed [AccBits-1:0] acc_o;

    neuron_dsp #(
        .InBits    (16),
        .WeightBits(16),
        .AccBits   (AccBits)
    ) dsp_inst (
        .clk_i      (clk_i),
        .rst_i      (rst_i),
        .en_i       (in_fire | busy_q),
        .load_bias_i(first_term),
        .data_i     (in_fire ? data_in_w   : data_w),
        .weight_i   (in_fire ? weight_in_w : weight_w),
        .bias_i     (AccBits'($signed(Bias))),
        .acc_o      (acc_o)
    );

    /* ------------------------- Output Activation ------------------------- */
    always_comb begin
        if (OutBits == 1) begin
            data_o = (acc_o > 0) ? OutBits'(1) : OutBits'(0);
        end else if (OutBits == 2) begin
            data_o = (acc_o > 0) ? 2'sb01 :
                     (acc_o < 0) ? 2'sb11 :
                                   2'sb00;
        end else begin
            data_o = OutBits'(acc_o);
        end
    end

endmodule
