import json
import math

def relu(x):
    return max(0.0, x)

def dot(x, w, b, out_sz):
    in_sz = len(x)
    y = [b[j] for j in range(out_sz)]
    for j in range(out_sz):
        for i in range(in_sz):
            y[j] += x[i] * w[i][j]
    return y

def fpu_exp(x):
    if x < -3.0:
        return 0.0
    if x > 88.0:
        x = 88.0
    # 1 + x + x^2/2 + x^3/6 + x^4/24
    return 1.0 + x + (x**2)/2.0 + (x**3)/6.0 + (x**4)/24.0

def softmax(x):
    mx = max(x)
    exps = [fpu_exp(v - mx) for v in x]
    sm = sum(exps)
    return [e / sm for e in exps]

def simulate_mips():
    with open("models/iris_model.json", "r") as f:
        model = json.load(f)
        
    x = [6.1, 2.8, 4.7, 1.2]
    
    # Layer 0
    l0 = model["layers"][0]
    w0 = l0["weights"]
    b0 = l0["biases"]
    y0 = dot(x, w0, b0, l0["output_size"])
    y0 = [relu(v) for v in y0]
    print(f"Layer 0 output: {y0}")
    
    # Layer 1
    l1 = model["layers"][1]
    w1 = l1["weights"]
    b1 = l1["biases"]
    y1 = dot(y0, w1, b1, l1["output_size"])
    print(f"Layer 1 pre-softmax: {y1}")
    
    prob = softmax(y1)
    print(f"MIPS Probabilities: {prob}")

simulate_mips()
