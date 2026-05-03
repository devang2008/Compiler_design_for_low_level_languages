import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import sys
import random

def main():
    print("=" * 60)
    print("Step 3: Weight Conversion")
    print("=" * 60)

    # Step 1 — Recreate model architecture
    class WineNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(11, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    model_path = Path("outputs/wine_model.pth")
    if not model_path.exists():
        print(f"[ERROR] {model_path} not found. Please run 02_train.py first.")
        sys.exit(1)

    model = WineNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Step 2 — Extract weights and biases
    extracted_layers = []
    layer_configs = [
        {"id": "layer_0", "name": "fc1", "in": 11, "out": 32, "act": "relu"},
        {"id": "layer_1", "name": "fc2", "in": 32, "out": 16, "act": "relu"},
        {"id": "layer_2", "name": "fc3", "in": 16, "out": 1,  "act": "sigmoid"}
    ]

    total_weights = 0
    weights_dict = {}

    for config in layer_configs:
        layer_module = getattr(model, config["name"])
        
        # PyTorch weights: [out_features, in_features]
        # Transpose to: [in_features, out_features] for deep_mips format
        weight_matrix = layer_module.weight.data.T.numpy().tolist()
        bias_vector = layer_module.bias.data.numpy().tolist()
        
        weights_dict[config["name"]] = {"W": weight_matrix, "b": bias_vector}
        
        w_shape = [len(weight_matrix), len(weight_matrix[0])]
        b_shape = [len(bias_vector)]
        w_min = float(np.min(weight_matrix))
        w_max = float(np.max(weight_matrix))
        b_min = float(np.min(bias_vector))
        b_max = float(np.max(bias_vector))
        
        print(f"{config['name']} weights: shape {w_shape}  min: {w_min:.3f}  max: {w_max:.3f}")
        print(f"{config['name']} biases:  shape {b_shape}      min: {b_min:.3f}  max: {b_max:.3f}")
        
        extracted_layers.append({
            "id": config["id"],
            "type": "Dense",
            "input_size": config["in"],
            "output_size": config["out"],
            "activation": config["act"],
            "weights": weight_matrix,
            "biases": bias_vector
        })
        total_weights += config["in"] * config["out"] + config["out"]

    # Step 3 — Validate extraction
    def pure_python_forward(x):
        # x is a list of 11 floats
        def relu(v):
            return max(0.0, v)
        def sigmoid(v):
            import math
            if v >= 0:
                return 1.0 / (1.0 + math.exp(-v))
            else:
                e = math.exp(v)
                return e / (1.0 + e)
                
        def linear(inp, w, b):
            out = list(b)
            for i in range(len(inp)):
                for j in range(len(b)):
                    out[j] += inp[i] * w[i][j]
            return out

        h1 = linear(x, weights_dict["fc1"]["W"], weights_dict["fc1"]["b"])
        h1 = [relu(v) for v in h1]
        
        h2 = linear(h1, weights_dict["fc2"]["W"], weights_dict["fc2"]["b"])
        h2 = [relu(v) for v in h2]
        
        h3 = linear(h2, weights_dict["fc3"]["W"], weights_dict["fc3"]["b"])
        out = [sigmoid(v) for v in h3]
        return out[0]

    # Generate a random input for validation
    test_inp = [random.random() for _ in range(11)]
    py_out = pure_python_forward(test_inp)
    
    with torch.no_grad():
        pt_out = model(torch.FloatTensor([test_inp])).item()
        
    max_diff = abs(py_out - pt_out)
    if max_diff > 1e-5:
        raise ValueError(f"Extraction error. PyTorch: {pt_out}, Python: {py_out}, Diff: {max_diff}")
    
    print(f"Weight extraction validated: max_diff = {max_diff:.6E}")

    # Step 4 — Select 10 test samples for MIPS embedding (5 positive, 5 negative predictions)
    data = np.load("outputs/train.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    with open("outputs/scaler_params.json") as f:
        scaler = json.load(f)
        
    means = scaler["means"]
    stds = scaler["stds"]

    test_samples_output = {"samples": []}
    pos_samples = []
    neg_samples = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            features_scaled = X_test[i].tolist()
            
            # Note: WineNet in 03_convert.py STILL has sigmoid in forward (line 27),
            # which is what we want for probability calculation here.
            out = model(torch.FloatTensor([features_scaled])).item()
            pred = 1 if out >= 0.5 else 0
            
            sample_data = {
                "index": i,
                "features_scaled": features_scaled,
                "features_raw": [features_scaled[j] * stds[j] + means[j] for j in range(11)],
                "python_probability": out,
                "python_prediction": pred,
                "true_label": int(y_test[i])
            }
            
            if pred == 1 and len(pos_samples) < 5:
                pos_samples.append(sample_data)
            elif pred == 0 and len(neg_samples) < 5:
                neg_samples.append(sample_data)
                
            if len(pos_samples) == 5 and len(neg_samples) == 5:
                break
                
    # Combine and sort
    selected = sorted(pos_samples + neg_samples, key=lambda x: x["index"])
    test_samples_output["samples"] = selected


    with open("outputs/test_samples.json", "w") as f:
        json.dump(test_samples_output, f, indent=2)

    # Step 5 — Build deep_mips JSON
    deep_mips_model = {
        "name": "wine_quality_binary",
        "input_shape": [11],
        "output_shape": [1],
        "quantize": False,
        "use_fpu": True,
        "layers": extracted_layers
    }

    with open("outputs/wine_model.json", "w") as f:
        json.dump(deep_mips_model, f, indent=2)

    # Step 6 — Print summary
    print("\nConversion complete.")
    print(f"Layers: 3 | Total weights: {total_weights}")
    print("10 test samples saved to outputs/test_samples.json")
    print("Model JSON saved to outputs/wine_model.json")

if __name__ == "__main__":
    main()
