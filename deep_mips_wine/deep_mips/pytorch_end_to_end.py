import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

print("1. Loading Iris dataset and training PyTorch model...")
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        # Note: We don't apply Softmax here because CrossEntropyLoss applies it internally during training.
        # But we will manually apply Softmax for the ground truth test sample.
        return x

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Quick training loop
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
for epoch in range(200):
    optimizer.zero_grad()
    out = model(X_train_t)
    loss = criterion(out, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()

# 2. Extract Test Sample
test_sample = X_test[0]
print(f"\n[Test Sample] Features: {test_sample}")

# Calculate Python Ground Truth Probabilities
with torch.no_grad():
    test_tensor = torch.tensor([test_sample])
    raw_logits = model(test_tensor)
    probs = torch.softmax(raw_logits, dim=1).numpy()[0]
    pred_class = np.argmax(probs)

print(f"[Python Ground Truth] Probabilities: {probs}")
print(f"[Python Ground Truth] Prediction: Class {pred_class}")

# 3. Save PyTorch Model
pth_path = "models/pytorch_iris.pth"
torch.save(model.state_dict(), pth_path)
print(f"\n2. Saved PyTorch model to {pth_path}...")

# 4. Convert to JSON
print("\n3. Converting .pth to .json using pth_to_json.py...")
subprocess.run(["python", "pth_to_json.py", pth_path])

json_path = "models/pytorch_iris.json"

# 5. Compile to MIPS Assembly
print("\n4. Compiling .json to MIPS .asm using main.py...")
subprocess.run(["python", "main.py", json_path, "--fpu"])

asm_path = "models/pytorch_iris.asm"

# 6. Inject Test Sample into MIPS assembly
print("\n5. Injecting test sample into MIPS assembly...")
with open(asm_path, "r") as f:
    asm_code = f.read()

# Replace the empty input buffer with our actual float values
input_str = ", ".join([str(val) for val in test_sample])
asm_code = asm_code.replace(
    "input_buffer:   .space  16",
    f"input_buffer:\n    .float {input_str}"
)

with open(asm_path, "w") as f:
    f.write(asm_code)

print("\nDone! Please run 'models/pytorch_iris.asm' in MARS.")
print("The printed output should closely match the [Python Ground Truth] Probabilities above!")
