import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def main():
    print("=" * 60)
    print("Step 1: Data Preparation")
    print("=" * 60)

    data_path = Path("data/winequality-red.csv")
    if not data_path.exists():
        print(f"[ERROR] Data file not found at {data_path}. Please provide it.")
        sys.exit(1)

    # Step 1 — Load CSV
    df = pd.read_csv(data_path, sep=';')
    print(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    print("First 3 rows:\n", df.head(3))
    print("\nLabel distribution before conversion:")
    print(df['quality'].value_counts().sort_index())

    # Step 2 — Validate
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values per column:")
        print(missing[missing > 0])
    
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"\nDropped {dup_count} duplicates.")
        df = df.drop_duplicates()
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Non-numeric values found in column {col}")

    # Step 3 — Convert labels
    df['quality'] = (df['quality'] >= 7).astype(int)
    print(f"Remaining samples: {len(df)}")
    val_counts = df['quality'].value_counts()
    print("\nNew label distribution:")
    print(f"0 = {val_counts.get(0, 0)} ({(val_counts.get(0, 0)/len(df))*100:.1f}%)")
    print(f"1 = {val_counts.get(1, 0)} ({(val_counts.get(1, 0)/len(df))*100:.1f}%)")

    # Step 4 — Separate features and labels
    feature_names = [c for c in df.columns if c != 'quality']
    features = df[feature_names].copy()
    labels = df['quality'].copy()

    # Step 5 — Compute and save scaler parameters
    means = []
    stds = []
    for col in feature_names:
        mean_i = features[col].mean()
        std_i = features[col].std(ddof=1)
        if std_i == 0:
            std_i = 1.0
        means.append(mean_i)
        stds.append(std_i)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    scaler_params = {
        "feature_names": feature_names,
        "means": means,
        "stds": stds
    }
    with open(outputs_dir / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    # Step 6 — Apply normalization
    for i, col in enumerate(feature_names):
        features[col] = (features[col] - means[i]) / stds[i]

    print("\nPer-feature stats after scaling (mean approx 0, std approx 1):")
    for col in feature_names:
        print(f"  {col}: mean={features[col].mean():.4f}, std={features[col].std(ddof=1):.4f}")

    # Step 7 — Train/val/test split
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    
    n_total = len(features)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    X = features.values
    y = labels.values
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Step 8 — Save
    np.savez(
        outputs_dir / "train.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names
    )

    # Step 9 — Print summary
    print("\nPreparation complete.")
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples | Test: {len(X_test)} samples")
    print(f"Positive class: {val_counts.get(1, 0)} ({(val_counts.get(1, 0)/len(df))*100:.1f}%) | Negative class: {val_counts.get(0, 0)} ({(val_counts.get(0, 0)/len(df))*100:.1f}%)")
    print("Scaler saved to outputs/scaler_params.json")
    print("Data saved to outputs/train.npz")

if __name__ == "__main__":
    main()
