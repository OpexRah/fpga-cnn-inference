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
    parameter PIPELINED  = 1;

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

    // Delayed signals for PIPELINED mode
    reg signed [ACC_WIDTH-1:0] acc_in_d;
    reg en_d;
    reg valid_in_d;
    reg signed [ACC_WIDTH-1:0] captured_acc_out;

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
        .en(PIPELINED ? en_d : en),
        .a(a),
        .b(b),
        .acc_in(PIPELINED ? acc_in_d : acc_in),
        .acc_out(acc_out),
        .valid_in(PIPELINED ? valid_in_d : valid_in),
        .valid_out(valid_out)
    );

    // Pipeline delay logic for PIPELINED mode
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_in_d <= {ACC_WIDTH{1'b0}};
            en_d <= 1'b0;
            valid_in_d <= 1'b0;
            captured_acc_out <= {ACC_WIDTH{1'b0}};
        end else begin
            // Delay acc_in, en, valid_in by one cycle for pipelined mode
            acc_in_d <= acc_in;
            en_d <= en;
            valid_in_d <= valid_in;
            
            // Always capture the latest acc_out
            captured_acc_out <= acc_out;
        end
    end

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
        // We'll do 4 multiply-accumulate steps sequentially
        // Example Q1.15 numbers: 0.5 -> 16384, -0.25 -> -8192, 0.1 -> 3277
        
        // Step 1: Start with acc = 0
        acc_ref = 0;
        mac_step(16'sd16384, 16'sd8192, acc_ref);    
        acc_ref = acc_ref + (16384 * 8192);
        
        // Wait for valid_out to be asserted and then wait one more cycle for pipelined result
        wait(valid_out);
        @(posedge clk);
        
        // Step 2: Use previous result 
        mac_step(-16'sd8192, 16'sd16384, captured_acc_out);   
        acc_ref = captured_acc_out + (-8192 * 16384);
        
        wait(valid_out);
        @(posedge clk);

        // Step 3: Use previous result
        mac_step(16'sd3277, -16'sd3277, captured_acc_out);    
        acc_ref = captured_acc_out + (3277 * -3277);
        
        wait(valid_out);
        @(posedge clk);

        // Step 4: Use previous result
        mac_step(16'sd16384, -16'sd16384, captured_acc_out);  
        acc_ref = captured_acc_out + (16384 * -16384);

        // --------------------------
        // Wait for pipeline latency
        // --------------------------
        repeat (8) @(posedge clk);  // Increased wait time for pipelined mode

        // --------------------------
        // Display results
        // --------------------------
        $display("Expected (reference) ACC result = %0d", acc_ref);
        $display("DUT output = %0d", PIPELINED ? captured_acc_out : acc_out);

        // Compare within tolerance (for fixed point rounding)
        if ($abs(acc_ref - (PIPELINED ? captured_acc_out : acc_out)) < 10)
            $display("✅ MAC TEST PASSED");
        else
            $display("❌ MAC TEST FAILED");

        $finish;
    end

endmodule
