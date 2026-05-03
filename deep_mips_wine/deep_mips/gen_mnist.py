"""Generate mnist_small.json with representative weights."""
import json, math, os

def gen_weights(rows, cols, seed=42):
    """Simple deterministic pseudo-random weights in [-0.5, 0.5]."""
    w = []
    v = seed
    for i in range(rows):
        row = []
        for j in range(cols):
            v = (v * 1103515245 + 12345) & 0x7FFFFFFF
            val = (v % 1000 - 500) / 1000.0
            row.append(round(val, 4))
        w.append(row)
    return w

def gen_biases(size, seed=7):
    v = seed
    b = []
    for i in range(size):
        v = (v * 1103515245 + 12345) & 0x7FFFFFFF
        val = (v % 200 - 100) / 1000.0
        b.append(round(val, 4))
    return b

model = {
    "name": "mnist_small",
    "input_shape": [784],
    "output_shape": [10],
    "quantize": False,
    "layers": [
        {
            "id": "layer_0",
            "type": "Dense",
            "input_size": 784,
            "output_size": 64,
            "activation": "relu",
            "weights": gen_weights(784, 64, seed=42),
            "biases": gen_biases(64, seed=7)
        },
        {
            "id": "layer_1",
            "type": "Dense",
            "input_size": 64,
            "output_size": 10,
            "activation": "softmax",
            "weights": gen_weights(64, 10, seed=99),
            "biases": gen_biases(10, seed=13)
        }
    ]
}

out = os.path.join(os.path.dirname(__file__), "models", "mnist_small.json")
with open(out, "w") as f:
    json.dump(model, f)
print(f"Generated {out} ({os.path.getsize(out)} bytes)")
