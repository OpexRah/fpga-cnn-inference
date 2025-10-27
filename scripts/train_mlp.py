import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os

# ========================
# Configuration
# ========================

OUTPUT_DIR = "../model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Network structure (must match FPGA implementation later)
INPUT_SIZE = 28 * 28
HIDDEN1 = 256
HIDDEN2 = 128
OUTPUT_SIZE = 10
EPOCHS = 5
BATCH_SIZE = 128
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Define Model
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
# Training
# ========================

def train():
    print(f"Training on {DEVICE}...")

    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    testset = datasets.MNIST("../data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    acc = 100.0 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

    # ========================
    # Export weights
    # ========================

    weights = {
        "fc1_weight": model.fc1.weight.detach().cpu().numpy(),
        "fc1_bias": model.fc1.bias.detach().cpu().numpy(),
        "fc2_weight": model.fc2.weight.detach().cpu().numpy(),
        "fc2_bias": model.fc2.bias.detach().cpu().numpy(),
        "fc3_weight": model.fc3.weight.detach().cpu().numpy(),
        "fc3_bias": model.fc3.bias.detach().cpu().numpy(),
    }

    for name, arr in weights.items():
        np.save(os.path.join(OUTPUT_DIR, f"{name}.npy"), arr)
    np.save(os.path.join(OUTPUT_DIR, "layer_shapes.npy"),
            np.array([INPUT_SIZE, HIDDEN1, HIDDEN2, OUTPUT_SIZE]))

    print(f"exported to {OUTPUT_DIR}/ (as .npy files)")
    print("Next: Run quantize_export.py to generate .coe files for FPGA.")

    return acc


if __name__ == "__main__":
    train()
