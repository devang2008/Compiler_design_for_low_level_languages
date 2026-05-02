import json
import os
import argparse

def pytorch_to_json(state_dict, name="pytorch_model"):
    """
    Heuristically extracts sequential Dense (Linear) layers from a PyTorch state_dict
    and formats them for Deep-MIPS.
    """
    # Group weights and biases by their common prefix
    # E.g., 'fc1.weight' and 'fc1.bias' -> 'fc1'
    layers_data = {}
    for key, tensor in state_dict.items():
        if not hasattr(tensor, 'numpy'):
            print(f"Skipping {key}: not a tensor.")
            continue
            
        parts = key.rsplit('.', 1)
        if len(parts) == 2 and parts[1] in ['weight', 'bias']:
            layer_name = parts[0]
            param_type = parts[1]
            
            if layer_name not in layers_data:
                layers_data[layer_name] = {}
            # PyTorch Linear weights are shape (out_features, in_features)
            # We need them transposed: (in_features, out_features) for Deep-MIPS
            if param_type == 'weight':
                # Convert to numpy, transpose, and convert to list
                layers_data[layer_name]['weights'] = tensor.detach().numpy().T.tolist()
            else:
                layers_data[layer_name]['biases'] = tensor.detach().numpy().tolist()

    if not layers_data:
        raise ValueError("No linear layer weights/biases found in the .pth file.")

    layers = []
    layer_names = list(layers_data.keys())
    
    # Heuristically assume the order in the state_dict is the forward order
    for i, layer_name in enumerate(layer_names):
        data = layers_data[layer_name]
        if 'weights' not in data or 'biases' not in data:
            print(f"Skipping {layer_name}: Missing either weights or biases.")
            continue
            
        weights = data['weights']
        biases = data['biases']
        
        in_size = len(weights)
        out_size = len(biases)
        
        # Heuristically guess activation:
        # Relu for hidden layers, Softmax for the final layer
        activation = "softmax" if i == len(layer_names) - 1 else "relu"

        layer_def = {
            "id": layer_name,
            "type": "Dense",
            "input_size": in_size,
            "output_size": out_size,
            "activation": activation,
            "weights": weights,
            "biases": biases
        }
        layers.append(layer_def)

    if not layers:
        raise ValueError("Failed to parse any valid layers.")

    input_shape = [layers[0]['input_size']]
    output_shape = [layers[-1]['output_size']]

    json_model = {
        "name": name,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "quantize": False,
        "layers": layers
    }
    
    return json_model

def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch .pth state_dict to Deep-MIPS .json")
    parser.add_argument("pth_file", help="Path to the .pth/.pt state_dict file")
    parser.add_argument("--out", help="Output .json path", default=None)
    args = parser.parse_args()

    print(f"Loading PyTorch model state_dict from {args.pth_file}...")
    
    try:
        import torch
    except ImportError:
        print("Error: 'torch' is not installed. Please install PyTorch to run this script.")
        print("Run: pip install torch")
        return

    # Load the state_dict (map_location='cpu' ensures it loads even if trained on GPU)
    try:
        state_dict = torch.load(args.pth_file, map_location='cpu', weights_only=True)
    except Exception as e:
        # Fallback if weights_only=True is not supported in older PyTorch
        state_dict = torch.load(args.pth_file, map_location='cpu')

    # If it's a full model object instead of a state_dict, try to extract the state_dict
    if hasattr(state_dict, 'state_dict'):
        print("Detected full model object. Extracting state_dict...")
        state_dict = state_dict.state_dict()

    base_name = os.path.splitext(os.path.basename(args.pth_file))[0]
    
    print("Extracting linear layers...")
    json_model = pytorch_to_json(state_dict, name=base_name)

    # Determine output path
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(args.pth_file), f"{json_model['name']}.json")

    # Save to JSON
    print("Saving JSON...")
    with open(out_path, "w") as f:
        json.dump(json_model, f, indent=2)
        
    print(f"\nSuccess! Deep-MIPS JSON saved to: {out_path}")
    print(f"Number of layers parsed: {len(json_model['layers'])}")
    print(f"You can now compile it using: python main.py {out_path} --fpu")

if __name__ == "__main__":
    main()
