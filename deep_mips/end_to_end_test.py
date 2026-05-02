import os
import subprocess
import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

print("1. Loading Iris dataset and training model...")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a small network: 4 inputs -> 8 hidden (ReLU) -> 3 outputs (Softmax)
model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# Take the first test sample
test_sample = X_test[0]
print(f"\n[Test Sample] Features: {test_sample}")
print(f"[Python Ground Truth] Probabilities: {model.predict_proba([test_sample])[0]}")
print(f"[Python Ground Truth] Prediction: Class {model.predict([test_sample])[0]}")

print("\n2. Saving model to iris_model.pkl...")
with open("models/iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n3. Converting .pkl to .json using pkl_to_json.py...")
subprocess.run(["python", "pkl_to_json.py", "models/iris_model.pkl"])

print("\n4. Compiling .json to MIPS .asm using main.py...")
subprocess.run(["python", "main.py", "models/iris_model.json", "--fpu"])

print("\n5. Injecting test sample into MIPS assembly...")
asm_path = "models/iris_model.asm"
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

print("\nDone! Please run 'iris_model.asm' in MARS. The printed output should match the [Python Ground Truth] Probabilities above!")
