import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import random
import sys

def main():
    print("=" * 60)
    print("Step 2: Model Training")
    print("=" * 60)

    # Step 1 — Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    # Step 2 — Load data
    train_npz_path = Path("outputs/train.npz")
    if not train_npz_path.exists():
        print("[ERROR] outputs/train.npz not found. Please run 01_prepare.py first.")
        sys.exit(1)
        
    data = np.load(train_npz_path)
    X_train = torch.FloatTensor(data["X_train"])
    y_train = torch.FloatTensor(data["y_train"]).unsqueeze(1)
    X_val = torch.FloatTensor(data["X_val"])
    y_val = torch.FloatTensor(data["y_val"]).unsqueeze(1)
    X_test = torch.FloatTensor(data["X_test"])
    y_test = torch.FloatTensor(data["y_test"]).unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Step 3 — Define model
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

    torch.manual_seed(42)
    model = WineNet()

    total_params = sum(p.numel() for p in model.parameters())
    print("WineNet: Layer names, shapes, total parameters")
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")
    print(f"Total parameters: {total_params}")

    # Step 4 — Define training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Step 5 — Training loop
    epochs = 200
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    patience_counter = 0
    early_stop_patience = 30
    best_model_path = Path("outputs/wine_model_best.pth")

    for epoch in range(1, epochs + 1):
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += inputs.size(0)
            
        train_loss_avg = train_loss / train_total
        train_acc = train_correct / train_total * 100

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += inputs.size(0)
                
        val_loss_avg = val_loss / val_total
        val_acc = val_correct / val_total * 100

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_acc  = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step(val_loss_avg)

    # Step 6 — Reload best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        test_loss = criterion(outputs, y_test).item()
        predictions = (outputs >= 0.5).float()
        
        # Calculate metrics
        TP = ((predictions == 1) & (y_test == 1)).sum().item()
        FP = ((predictions == 1) & (y_test == 0)).sum().item()
        TN = ((predictions == 0) & (y_test == 0)).sum().item()
        FN = ((predictions == 0) & (y_test == 1)).sum().item()
        
        test_acc = (TP + TN) / (TP + FP + TN + FN) * 100
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n+" + "-" * 46 + "+")
    print("| Test Results                                 |")
    print(f"| Loss:      {test_loss:<34.4f}|")
    print(f"| Accuracy:  {test_acc:<33.2f}%|")
    print(f"| Precision: {precision:<34.4f}|")
    print(f"| Recall:    {recall:<34.4f}|")
    print(f"| F1 Score:  {f1_score:<34.4f}|")
    print("+" + "-" * 46 + "+")

    # Step 7 — Save final model
    final_model_path = Path("outputs/wine_model.pth")
    torch.save(model.state_dict(), final_model_path)

    model_info = {
        "architecture": "WineNet",
        "layers": [
            {"name": "fc1", "in": 11, "out": 32, "activation": "relu"},
            {"name": "fc2", "in": 32, "out": 16, "activation": "relu"},
            {"name": "fc3", "in": 16, "out": 1,  "activation": "sigmoid"}
        ],
        "test_accuracy": round(test_acc, 2),
        "f1_score": round(f1_score, 4),
        "total_params": total_params,
        "training_seed": 42,
        "epochs_trained": epoch
    }
    
    with open("outputs/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    # Step 8 - Pick 10 test samples and generate outputs/test_samples.json
    print("\n+" + "-" * 46 + "+")
    print("| Test Sample Details                          |")
    print("+" + "-" * 46 + "+")
    print("| Idx | Prob   | Pred | True | Raw Feature 0   |")
    print("+" + "-" * 46 + "+")
    
    test_samples_json = {"samples": []}
    
    # We need X_test_raw. We can extract it from the npz or the dataset if we had raw.
    # Wait, we don't have X_test_raw easily accessible because train.npz only has X_test.
    # We will just print the scaled feature 0.
    for i in range(10):
        feature_scaled = X_test[i].unsqueeze(0)
        true_label     = int(y_test[i].item())
        
        prob = model(feature_scaled).item()
        pred = 1 if prob >= 0.5 else 0
        
        print(f"| {i:3d} | {prob:.4f} |  {pred}   |  {true_label}   | {feature_scaled[0][0]:.4f}          |")
        
        test_samples_json["samples"].append({
            "features_raw": [], # not available in this script
            "features_scaled": feature_scaled[0].tolist(),
            "python_probability": prob,
            "python_prediction": pred,
            "true_label": true_label
        })
    print("+" + "-" * 46 + "+")

    with open("outputs/test_samples.json", "w") as f:
        json.dump(test_samples_json, f, indent=2)

    # Step 9 — Print final summary
    print("\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Model saved to {final_model_path}")

if __name__ == "__main__":
    main()
