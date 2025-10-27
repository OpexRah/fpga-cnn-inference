`timescale 1ns / 1ps
// tb_mac_unit.v
// Simple testbench for mac_unit.v
//
// Tests multiply-accumulate behavior for signed fixed-point numbers.
// Prints results to console for verification.

module tb_mac_unit();

    // ==============================
    // Parameters
    // ==============================
    parameter DATA_WIDTH = 16;
    parameter ACC_WIDTH  = 40;
    parameter CLK_PERIOD = 10; // 100 MHz
    parameter PIPELINED  = 0;

    // ==============================
    // Signals
    // ==============================
    reg clk;
    reg rst_n;
    reg en;
    reg valid_in;
    reg signed [DATA_WIDTH-1:0] a;
    reg signed [DATA_WIDTH-1:0] b;
    reg signed [ACC_WIDTH-1:0] acc_in;
    wire signed [ACC_WIDTH-1:0] acc_out;
    wire valid_out;

    // ==============================
    // DUT
    // ==============================
    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .PIPELINED(PIPELINED)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .en(en),
        .a(a),
        .b(b),
        .acc_in(acc_in),
        .acc_out(acc_out),
        .valid_in(valid_in),
        .valid_out(valid_out)
    );

    // ==============================
    // Clock generation
    // ==============================
    initial clk = 0;
    always #(CLK_PERIOD/2) clk = ~clk;

    // ==============================
    // Task for one MAC operation
    // ==============================
    task mac_step;
        input signed [DATA_WIDTH-1:0] ta;
        input signed [DATA_WIDTH-1:0] tb;
        input signed [ACC_WIDTH-1:0]  tacc_in;
        begin
            @(negedge clk);
            a        <= ta;
            b        <= tb;
            acc_in   <= tacc_in;
            en       <= 1'b1;
            valid_in <= 1'b1;
            @(negedge clk);
            en       <= 1'b0;
            valid_in <= 1'b0;
        end
    endtask

    // ==============================
    // Test sequence
    // ==============================
    integer i;
    reg signed [ACC_WIDTH-1:0] acc_ref;

    initial begin
        // Initialize
        $display("=== MAC UNIT TEST START ===");
        a = 0;
        b = 0;
        acc_in = 0;
        en = 0;
        valid_in = 0;
        rst_n = 0;
        acc_ref = 0;

        // Reset
        repeat (2) @(posedge clk);
        rst_n = 1;

        // --------------------------
        // Apply a few test vectors
        // --------------------------
        // We'll do 4 multiply-accumulate steps
        // Example Q1.15 numbers: 0.5 -> 16384, -0.25 -> -8192, 0.1 -> 3277
        mac_step(16'sd16384, 16'sd8192, acc_ref);    // 0.5 * 0.25 = 0.125
        acc_ref = acc_ref + (16384 * 8192);

        mac_step(-16'sd8192, 16'sd16384, acc_ref);   // -0.25 * 0.5 = -0.125
        acc_ref = acc_ref + (-8192 * 16384);

        mac_step(16'sd3277, -16'sd3277, acc_ref);    // 0.1 * -0.1 = -0.01
        acc_ref = acc_ref + (3277 * -3277);

        mac_step(16'sd16384, -16'sd16384, acc_ref);  // 0.5 * -0.5 = -0.25
        acc_ref = acc_ref + (16384 * -16384);

        // --------------------------
        // Wait for pipeline latency
        // --------------------------
        repeat (5) @(posedge clk);

        // --------------------------
        // Display results
        // --------------------------
        $display("Expected (reference) ACC result = %0d", acc_ref);
        $display("DUT output = %0d", acc_out);

        // Compare within tolerance (for fixed point rounding)
        if ($abs(acc_ref - acc_out) < 10)
            $display("✅ MAC TEST PASSED");
        else
            $display("❌ MAC TEST FAILED");

        $finish;
    end

endmodule
