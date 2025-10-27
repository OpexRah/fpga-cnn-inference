import torch
import numpy as np
from train_quantize import LeNet

def quantize_and_export(model_path="lenet_mnist.pth", out_dir="./weights/", bits=8):
    model = LeNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    def quantize(tensor):
        scale = 2 ** (bits - 1) - 1
        q = torch.clamp((tensor * scale).round(), -scale, scale)
        return q.numpy().astype(np.int8), 1.0 / scale

    scales = {}

    for name, param in model.named_parameters():
        arr, scale = quantize(param.data)
        np.ascontiguousarray(arr).tofile(f"{out_dir}/{name.replace('.', '_')}.bin")
        scales[name] = scale

    import json
    with open(f"{out_dir}/quant_scale.json", "w") as f:
        json.dump(scales, f, indent=2)

    print("Quantized weights exported to", out_dir)

if __name__ == "__main__":
    quantize_and_export()
