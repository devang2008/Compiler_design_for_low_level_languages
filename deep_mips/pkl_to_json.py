import json
import os
import argparse
import pickle
import numpy as np

def sklearn_mlp_to_json(model, name="exported_model"):
    """
    Extracts weights and biases from a scikit-learn MLPClassifier
    and converts them to the Deep-MIPS JSON format.
    """
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise ValueError("Model does not appear to be a trained MLPClassifier.")

    layers = []
    
    # MLPClassifier attributes:
    # coefs_ is a list of weight matrices. shape: (input_size, output_size)
    # intercepts_ is a list of bias vectors. shape: (output_size,)
    
    input_shape = [model.coefs_[0].shape[0]]
    output_shape = [model.coefs_[-1].shape[1]]
    
    for i in range(len(model.coefs_)):
        weights = model.coefs_[i].tolist() # Already Input x Output shape
        biases = model.intercepts_[i].tolist()
        
        in_size = len(weights)
        out_size = len(biases)
        
        # Determine activation
        if i == len(model.coefs_) - 1:
            # Last layer
            activation = "softmax" if model.out_activation_ == "softmax" else model.out_activation_
        else:
            # Hidden layer
            activation = model.activation
            
        layer_def = {
            "id": f"layer_{i}",
            "type": "Dense",
            "input_size": in_size,
            "output_size": out_size,
            "activation": activation,
            "weights": weights,
            "biases": biases
        }
        layers.append(layer_def)

    json_model = {
        "name": name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "quantize": False,
        "layers": layers
    }
    
    return json_model

def main():
    parser = argparse.ArgumentParser(description="Convert a .pkl model to Deep-MIPS .json")
    parser.add_argument("pkl_file", help="Path to the .pkl model file")
    parser.add_argument("--out", help="Output .json path (default: auto-generated)", default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.pkl_file}...")
    
    # Load the pickle file
    with open(args.pkl_file, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Model type: {type(model).__name__}")
    
    # Determine the framework and extract
    if "sklearn" in str(type(model)) and "MLPClassifier" in str(type(model)):
        print("Detected Scikit-Learn MLPClassifier. Extracting weights...")
        base_name = os.path.splitext(os.path.basename(args.pkl_file))[0]
        json_model = sklearn_mlp_to_json(model, name=base_name)
    else:
        print("Error: Unsupported model format.")
        print("This script currently supports: Scikit-Learn MLPClassifier.")
        print("For PyTorch models (.pth/.pt), please write a custom extraction script using torch.load().")
        return

    # Determine output path
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.pkl_file), f"{json_model['name']}.json")

    # Save to JSON
    with open(out_path, "w") as f:
        json.dump(json_model, f, indent=2)
        
    print(f"\nSuccess! Deep-MIPS JSON saved to: {out_path}")
    print(f"You can now compile it using: python main.py {out_path} --fpu")

if __name__ == "__main__":
    main()
