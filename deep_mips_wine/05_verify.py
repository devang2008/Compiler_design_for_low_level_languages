import json
import math
from pathlib import Path
import sys

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (1.0 + e)

def relu(x):
    return max(0.0, x)

def matmul_bias(inputs, weights, biases):
    # weights shape: [input_size][output_size]
    output_size = len(biases)
    input_size  = len(inputs)
    output = list(biases)   # start with bias
    for i in range(input_size):
        for j in range(output_size):
            output[j] += inputs[i] * weights[i][j]
    return output

def forward_pass(model_json, input_features):
    x = input_features
    for layer in model_json["layers"]:
        x = matmul_bias(x, layer["weights"], layer["biases"])
        act = layer["activation"]
        if act == "relu":
            x = [relu(v) for v in x]
        elif act == "sigmoid":
            x = [sigmoid(v) for v in x]
    return x   # list of output values

def main():
    print("=" * 60)
    print("Step 5: Verification Report")
    print("=" * 60)

    test_samples_path = Path("outputs/test_samples.json")
    model_json_path = Path("outputs/wine_model.json")
    
    if not test_samples_path.exists() or not model_json_path.exists():
        print("[ERROR] Required JSON files not found in outputs/.")
        sys.exit(1)

    with open(test_samples_path) as f:
        test_samples = json.load(f)
        
    with open(model_json_path) as f:
        model_json = json.load(f)

    print("Self-check (PyTorch vs Pure Python):")
    
    results = []
    
    for i, sample in enumerate(test_samples["samples"]):
        features = sample["features_scaled"]
        output = forward_pass(model_json, features)
        prob = output[0]
        pred = 1 if prob >= 0.5 else 0
        expected_pred = sample["python_prediction"]
        expected_prob = sample["python_probability"]
        
        diff = abs(expected_prob - prob)
        
        if diff <= 1e-4:
            print(f"  Sample {i}: PyTorch={expected_prob:.4f}  PurePy={prob:.4f}  diff={diff:.4f}  OK")
        else:
            print(f"  Sample {i}: PyTorch={expected_prob:.4f}  PurePy={prob:.4f}  diff={diff:.4f}  FAIL")
            raise ValueError("Forward pass mismatch - check 03_convert.py")
            
        match = (pred == expected_pred)
        
        results.append({
            "sample_index": i,
            "features_raw": sample["features_raw"],
            "python_probability_original": expected_prob,
            "python_probability_recomputed": prob,
            "prediction": pred,
            "true_label": expected_pred,
            "match": match
        })

    report_lines = []
    report_lines.append("========================================================")
    report_lines.append("Deep-MIPS Wine Quality - Verification Report")
    report_lines.append("========================================================")
    report_lines.append("")
    report_lines.append("Sample | True | Py Prob | Py Pred | MIPS Pred | Match")
    report_lines.append("-------+------+---------+---------+-----------+-------")
    
    mips_sequence = []
    for r in results:
        i = r["sample_index"]
        true_label = r["true_label"]
        py_prob = r["python_probability_recomputed"]
        py_pred = r["prediction"]
        mips_sequence.append(str(py_pred))
        
        report_lines.append(f"  {i:2d}   |  {true_label}   |  {py_prob:.4f} |    {py_pred}    |    ?      |  ?")

    expected_str = " ".join(mips_sequence)
    
    print("\n" + "\n".join(report_lines))
    print(f"\nExpected MIPS output sequence: {expected_str}")
    print("Run outputs/wine_model.asm in MARS and compare.")
    print("Each line printed by MARS should match the sequence above.")

    # Write to file
    out_file = Path("outputs/verification_report.txt")
    with open(out_file, "w") as f:
        f.write("Expected MIPS output (10 lines):\n")
        for digit in mips_sequence:
            f.write(f"{digit}\n")
        f.write("\n")
        f.write("\n".join(report_lines))
        f.write("\n")
        
    print(f"\nVerification report written to {out_file}")
    print(f"Expected MIPS sequence: {expected_str}")
    print("Run outputs/wine_model.asm in MARS simulator.")
    print("MARS should print 10 lines, each 0 or 1.")
    print("Compare MARS output to expected sequence above.")

if __name__ == "__main__":
    main()
