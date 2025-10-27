import torch
from torchvision import datasets, transforms
from train_quantize import LeNet
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
image, label = testset[0]

# Save input image as binary
image.numpy().astype(np.float32).tofile("test_vectors/input_image_0.bin")

# Save expected output
model = LeNet()
model.load_state_dict(torch.load("lenet_mnist.pth", map_location="cpu"))
model.eval()
out = model(image.unsqueeze(0))
out.detach().numpy().astype(np.float32).tofile("test_vectors/golden_output_0.bin")

print(f"Saved test vector (label={label})")
