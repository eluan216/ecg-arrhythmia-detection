"""Train PyTorch 1D-CNN on ECG beats."""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. CNN training disabled.")

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

if TORCH_AVAILABLE:
    from src.data.synthetic import generate_synthetic_dataset
    from src.models.cnn_pytorch import ECG1DCNN


def load_or_generate_data(
    beat_data_file: Optional[str] = None,
    beat_data_dir: str = "data/processed",
    use_synthetic: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load beat data from file or generate synthetic data.

    Args:
        beat_data_file: Path to specific NPZ file with beats.
        beat_data_dir: Directory to look for beat files.
        use_synthetic: If no file found, generate synthetic data.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    beat_data_dir = Path(beat_data_dir)

    # Try to load from specific file or default locations
    if beat_data_file and Path(beat_data_file).exists():
        data = np.load(beat_data_file)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"Loaded data from {beat_data_file}")
        return X_train, y_train, X_test, y_test

    # Try default synthetic file
    synthetic_file = beat_data_dir / "synthetic_beats.npz"
    if synthetic_file.exists():
        data = np.load(synthetic_file)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"Loaded synthetic data from {synthetic_file}")
        return X_train, y_train, X_test, y_test

    # Generate synthetic data
    if use_synthetic:
        print("Generating synthetic training data...")
        beat_data_dir.mkdir(parents=True, exist_ok=True)
        X_train, y_train, X_test, y_test = generate_synthetic_dataset(
            n_samples_per_class=200,
            beat_length=256,
            test_ratio=0.2,
        )
        return X_train, y_train, X_test, y_test

    raise FileNotFoundError(
        f"No beat data found at {beat_data_dir} and synthetic generation disabled."
    )


def train_cnn(
    beat_data_file: Optional[str] = None,
    beat_data_dir: str = "data/processed",
    output_dir: str = "models",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_synthetic: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Train and evaluate the PyTorch 1D-CNN model.

    Args:
        beat_data_file: Optional path to specific data file.
        beat_data_dir: Directory containing preprocessed beat data.
        output_dir: Directory to save trained model and metrics.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for Adam optimizer.
        use_synthetic: Generate synthetic data if real data not found.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with training and evaluation metrics.
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping CNN training.")
        return {"status": "torch_not_available"}

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    try:
        X_train, y_train, X_test, y_test = load_or_generate_data(
            beat_data_file=beat_data_file,
            beat_data_dir=beat_data_dir,
            use_synthetic=use_synthetic,
        )
    except FileNotFoundError as e:
        print(f"Data loading error: {e}")
        return {"status": "data_not_found", "error": str(e)}

    print(f"Training data shape: {X_train.shape}, labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels shape: {y_test.shape}")

    # Ensure data is float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model
    model = ECG1DCNN(num_classes=5, dropout_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Class weights to handle imbalance
    class_counts = np.bincount(y_train)
    class_weights = torch.from_numpy(
        len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    # Training loop
    print(f"Training for {num_epochs} epochs...")
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for beats, batch_labels in train_loader:
            beats = beats.unsqueeze(1).to(device)  # Add channel dim: (batch, 1, seq_len)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(beats)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_labels)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Evaluate
    print("Evaluating on test set...")
    model.eval()
    y_pred = []
    y_test_list = []
    y_proba = []

    with torch.no_grad():
        for beats, batch_labels in test_loader:
            beats = beats.unsqueeze(1).to(device)
            logits = model(beats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_test_list.extend(batch_labels.numpy())
            y_proba.extend(probs.cpu().numpy())

    y_pred = np.array(y_pred)
    y_test_arr = np.array(y_test_list)
    y_proba = np.array(y_proba)

    # Metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_arr, y_pred, average=None
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_test_arr, y_pred, average="macro"
    )
    cm = confusion_matrix(y_test_arr, y_pred)

    metrics = {
        "model": "1D-CNN (PyTorch)",
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": cm.tolist(),
        "class_labels": ["N", "V", "S", "F", "Q"],
    }

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ecg-cnn")
        with mlflow.start_run(run_name="1d_cnn"):
            mlflow.log_params(
                {
                    "model": "1D-CNN",
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "device": str(device),
                }
            )
            mlflow.log_metrics(
                {
                    "macro_f1": macro_f1,
                    "macro_precision": macro_prec,
                    "macro_recall": macro_rec,
                }
            )
            for i, cls in enumerate(["N", "V", "S", "F", "Q"]):
                mlflow.log_metrics(
                    {
                        f"f1_{cls}": float(f1[i]),
                        f"precision_{cls}": float(precision[i]),
                        f"recall_{cls}": float(recall[i]),
                        f"support_{cls}": int(support[i]),
                    }
                )

    # Save metrics and model
    metrics_file = output_dir / "cnn_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    model_file = output_dir / "cnn_model.pth"
    torch.save(model.state_dict(), model_file)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}/")
    print(f"{'='*60}")
    print(f"Macro Precision: {macro_prec:.4f}")
    print(f"Macro Recall:    {macro_rec:.4f}")
    print(f"Macro F1-Score:  {macro_f1:.4f}")
    print("\nPer-class metrics (N, V, S, F, Q):")
    print(f"  Precision: {[f'{p:.3f}' for p in precision]}")
    print(f"  Recall:    {[f'{r:.3f}' for r in recall]}")
    print(f"  F1-Score:  {[f'{sc:.3f}' for sc in f1]}")
    print(f"  Support:   {support}")

    return metrics


if __name__ == "__main__":
    train_cnn()
