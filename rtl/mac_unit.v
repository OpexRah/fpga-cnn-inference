// mac_unit.v
// Parameterized fixed-point multiply-accumulate unit.
// - Signed inputs (two's complement).
// - Multiplies `A` and `B` -> product width = 2*DATA_WIDTH.
// - Adds product to accumulator (ACC_WIDTH).
// - Optional 1-stage pipeline between multiplier and adder.
//
// Notes:
// - This module does not perform saturation by default on the accumulator
//   (overflow wraps according to two's-complement arithmetic).
// - To infer DSP slices in Vivado, leave multipliers natural and let
//   synthesis infer DSP48E. If you want explicit DSP instantiation,
//   replace the multiplication with a vendor primitive.
//
// Q-format handling:
// - This MAC treats inputs as integers. The user is responsible for
//   interpreting them as fixed-point (e.g., Q1.15) and for adjusting
//   accumulator scaling externally if needed.

`timescale 1ns / 1ps
module mac_unit #(
    parameter DATA_WIDTH = 16,       // input A/B bit width (signed)
    parameter ACC_WIDTH  = 40,       // accumulator width (signed) - should be >= (2*DATA_WIDTH + log2(#accum_terms))
    parameter PIPELINED  = 1         // 0 = no pipeline, 1 = one-stage pipeline between multiply & add
)(
    input  wire                          clk,
    input  wire                          rst_n,     // active-low sync reset
    input  wire                          en,        // enable (when high, consume inputs)
    input  wire signed [DATA_WIDTH-1:0]  a,
    input  wire signed [DATA_WIDTH-1:0]  b,
    input  wire signed [ACC_WIDTH-1:0]   acc_in,    // previous accumulator (or 0 to start)
    output reg  signed [ACC_WIDTH-1:0]   acc_out,
    input  wire                          valid_in,  // indicates a,b,acc_in are valid
    output reg                           valid_out  // indicates acc_out is valid
);

    // Derived widths
    localparam PROD_WIDTH = 2 * DATA_WIDTH;
    // Internal signals
    reg signed [PROD_WIDTH-1:0] prod;
    reg signed [PROD_WIDTH-1:0] prod_pipe; // if pipelined
    reg signed [ACC_WIDTH-1:0]  add_result;

    // Multiply (combinational; inferred DSP)
    // We register product optionally to create a pipeline stage.
    always @(*) begin
        prod = $signed(a) * $signed(b); // PROD_WIDTH bits
    end

    // Pipelining & valid gating
    // Optionally register product to separate multiply and add stages.
    generate
        if (PIPELINED) begin : gen_pipelined
            always @(posedge clk) begin
                if (!rst_n) begin
                    prod_pipe <= {PROD_WIDTH{1'b0}};
                    valid_out <= 1'b0;
                    add_result <= {ACC_WIDTH{1'b0}};
                end else begin
                    if (en && valid_in) begin
                        prod_pipe <= prod;
                        // perform addition next cycle using previous acc_in
                        // we register valid to indicate output will be ready next cycle
                        valid_out <= 1'b1;
                    end else begin
                        valid_out <= 1'b0;
                        prod_pipe <= {PROD_WIDTH{1'b0}};
                    end

                    // Extend product to accumulator width (sign-extend)
                    // then add to acc_in (note: acc_in should be stable/registered by caller)
                    // We perform the add combinationally here and register the result to acc_out.
                    // Since product was registered, pipeline separation is achieved.
                    add_result <= $signed({{(ACC_WIDTH-PROD_WIDTH){prod_pipe[PROD_WIDTH-1]}}, prod_pipe}) + acc_in;
                    acc_out <= add_result;
                end
            end
        end else begin : gen_nonpipelined
            always @(posedge clk) begin
                if (!rst_n) begin
                    acc_out <= {ACC_WIDTH{1'b0}};
                    valid_out <= 1'b0;
                end else begin
                    if (en && valid_in) begin
                        // sign-extend product into accumulator width and add immediately
                        acc_out <= $signed({{(ACC_WIDTH-PROD_WIDTH){prod[PROD_WIDTH-1]}}, prod}) + acc_in;
                        valid_out <= 1'b1;
                    end else begin
                        valid_out <= 1'b0;
                    end
                end
            end
        end
    endgenerate

endmodule
