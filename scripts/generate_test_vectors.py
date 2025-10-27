#!/usr/bin/env python3
"""
generate_test_vectors.py
Generate quantized input vectors and golden outputs for FPGA verification.
Uses saved .npy weights and scales.json from quantized export.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import json
import os

# ========================
# Configuration
# ========================

MODEL_DIR = "../model"
SCALE_JSON_PATH = os.path.join(MODEL_DIR, "quantized/scales.json")
OUTPUT_DIR = os.path.join(MODEL_DIR, "test_vectors")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_SIZE = 28 * 28
HIDDEN1 = 256
HIDDEN2 = 128
OUTPUT_SIZE = 10
NUM_SAMPLES = 10
QFRAC = 15
BITS = 16

# ========================
# Model Definition (must match training)
# ========================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN2, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# ========================
# Helpers
# ========================

def write_mem(filename, data):
    """Write 1D array to Verilog .mem file ($readmemh compatible)."""
    with open(filename, "w") as f:
        for val in data.flatten():
            f.write(f"{(int(val) & 0xFFFF):04x}\n")

def float_to_fixed(x, qfrac=15, bits=16):
    """Convert float → signed fixed-point (saturation)."""
    scale = 2 ** qfrac
    x_q = np.round(x * scale).astype(np.int64)
    min_val = -(2 ** (bits - 1))
    max_val = (2 ** (bits - 1)) - 1
    x_q = np.clip(x_q, min_val, max_val)
    return x_q.astype(np.int16)

def load_mnist_samples(num_samples=10):
    """Load a few MNIST test samples as flattened numpy arrays."""
    dataset = datasets.MNIST(
        root="../data",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    imgs, labels = [], []
    for i in range(num_samples):
        img, label = dataset[i]
        imgs.append(img.view(-1).numpy())
        labels.append(label)
    return np.stack(imgs), np.array(labels)


# ========================
# Main Generation
# ========================

def generate_test_vectors():
    print("Rebuilding MLP and loading trained weights...")

    # Load model
    model = MLP()
    model.eval()

    # Load .npy weights
    model.fc1.weight.data = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc1_weight.npy")), dtype=torch.float32)
    model.fc1.bias.data   = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc1_bias.npy")), dtype=torch.float32)
    model.fc2.weight.data = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc2_weight.npy")), dtype=torch.float32)
    model.fc2.bias.data   = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc2_bias.npy")), dtype=torch.float32)
    model.fc3.weight.data = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc3_weight.npy")), dtype=torch.float32)
    model.fc3.bias.data   = torch.tensor(np.load(os.path.join(MODEL_DIR, "fc3_bias.npy")), dtype=torch.float32)

    print("Weights loaded from .npy files.")

    # Load quantization scales
    with open(SCALE_JSON_PATH, "r") as f:
        scales = json.load(f)
    print(f"Loaded quantization scales from {SCALE_JSON_PATH}")

    # Load MNIST samples
    x_test, y_test = load_mnist_samples(NUM_SAMPLES)
    print(f"Loaded {NUM_SAMPLES} MNIST samples")

    # Quantize input images (Q1.15)
    x_test_q = float_to_fixed(x_test, qfrac=QFRAC, bits=BITS)

    # Run forward pass (float)
    with torch.no_grad():
        outputs = model(torch.tensor(x_test, dtype=torch.float32))
        outputs_np = outputs.numpy()

    # Quantize golden outputs (use fc3 scale)
    last_layer = "fc3_weight"
    qfrac_out = scales[last_layer]["frac_bits"]
    bits_out = scales[last_layer]["bits"]
    y_golden_q = float_to_fixed(outputs_np, qfrac=qfrac_out, bits=bits_out)

    # Save results
    np.save(os.path.join(OUTPUT_DIR, "input_vectors.npy"), x_test_q)
    np.save(os.path.join(OUTPUT_DIR, "golden_outputs.npy"), y_golden_q)
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), y_test)

    write_mem(os.path.join(OUTPUT_DIR, "input_vectors.mem"), x_test_q.flatten())
    write_mem(os.path.join(OUTPUT_DIR, "golden_outputs.mem"), y_golden_q.flatten())

    print("\nTest vector generation complete.")
    print(f"→ Inputs saved to: {OUTPUT_DIR}/input_vectors.mem")
    print(f"→ Golden outputs saved to: {OUTPUT_DIR}/golden_outputs.mem")
    print(f"→ Labels saved to: {OUTPUT_DIR}/labels.npy")

    return x_test_q, y_golden_q, y_test


# ========================
# Auto-run
# ========================

if __name__ == "__main__":
    generate_test_vectors()
