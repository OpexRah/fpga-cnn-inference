import numpy as np
import os
import json

# ========================
# Configuration Defaults
# ========================

DEFAULT_INPUT_DIR = "../model"
DEFAULT_OUTPUT_DIR = "../model/quantized"
DEFAULT_QFRAC = 15     # fractional bits (for Q1.15)
DEFAULT_BITS = 16      # total bits


# ========================
# Helper Functions
# ========================

def float_to_fixed(x, total_bits=16, frac_bits=15):
    """Convert float numpy array to signed fixed-point integer array."""
    scale = 2 ** frac_bits
    x_q = np.round(x * scale).astype(np.int64)
    # Saturate
    min_val = -(2 ** (total_bits - 1))
    max_val = (2 ** (total_bits - 1)) - 1
    x_q = np.clip(x_q, min_val, max_val)
    return x_q, scale


def write_coe(filename, data):
    """Write data (1D array) to Vivado .coe format."""
    with open(filename, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        hex_data = [f"{(int(x) & 0xFFFF):04x}" for x in data.flatten()]
        f.write(",\n".join(hex_data))
        f.write(";\n")


def write_mem(filename, data):
    """Write data (1D array) to Verilog .mem format (for $readmemh)."""
    with open(filename, "w") as f:
        for val in data.flatten():
            f.write(f"{(int(val) & 0xFFFF):04x}\n")


# ========================
# Main Export Function
# ========================

def quantize_and_export(
    input_dir=DEFAULT_INPUT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    bits=DEFAULT_BITS,
    qfrac=DEFAULT_QFRAC,
    scale_json_name="scales.json"
):
    """
    Quantize all .npy weight/bias files in input_dir and export to output_dir.
    Also save layer-wise scale factors to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    layer_files = [f for f in os.listdir(input_dir)
                   if f.endswith(".npy") and ("weight" in f or "bias" in f)]

    if not layer_files:
        print(f"No .npy weight/bias files found in {input_dir}.")
        return {}

    print(f"Quantizing {len(layer_files)} layers from {input_dir} ...")

    scales = {}
    for fname in sorted(layer_files):
        path = os.path.join(input_dir, fname)
        arr_f = np.load(path)
        arr_q, scale = float_to_fixed(arr_f, bits, qfrac)

        base = fname.replace(".npy", "")
        coe_path = os.path.join(output_dir, base + ".coe")
        mem_path = os.path.join(output_dir, base + ".mem")
        npy_path = os.path.join(output_dir, base + f"_q{bits}.npy")

        np.save(npy_path, arr_q.astype(np.int16))
        write_coe(coe_path, arr_q)
        write_mem(mem_path, arr_q)

        scales[base] = {
            "scale": scale,
            "bits": bits,
            "frac_bits": qfrac,
            "min_float": float(arr_f.min()),
            "max_float": float(arr_f.max())
        }

        print(f"→ {fname:20s} | "
              f"range [{arr_f.min():+.3f}, {arr_f.max():+.3f}] "
              f"→ int16 [{arr_q.min()}, {arr_q.max()}], scale=2^{qfrac}")

    # Save all scales to JSON
    json_path = os.path.join(output_dir, scale_json_name)
    with open(json_path, "w") as jf:
        json.dump(scales, jf, indent=4)

    print(f"\nQuantization done. Scale factors saved to {json_path}")
    return scales


# ========================
# Auto-run Section
# ========================

if __name__ == "__main__":
    # Run with default directories
    quantize_and_export()
