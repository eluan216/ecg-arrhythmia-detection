"""Train PyTorch 1D-CNN on ECG beats.

Default is fail-closed: real beat data required unless --allow-synthetic is set.
CNN inter-patient MIT-BIH results are not yet in the README measured table.
"""

from __future__ import annotations

import argparse
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

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if TORCH_AVAILABLE:
    from src.data.synthetic import generate_synthetic_dataset
    from src.models.cnn_pytorch import ECG1DCNN


def load_or_generate_data(
    beat_data_file: Optional[str] = None,
    beat_data_dir: str = "data/processed",
    use_synthetic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beat_data_dir = Path(beat_data_dir)
    if beat_data_file and Path(beat_data_file).exists():
        data = np.load(beat_data_file)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    for candidate in (
        beat_data_dir / "beats.npz",
        beat_data_dir / "mitbih_beats.npz",
        beat_data_dir / "synthetic_beats.npz",
    ):
        if candidate.exists():
            data = np.load(candidate)
            if "synthetic" in candidate.name:
                print("Note: file is synthetic — do not publish as MIT-BIH metrics.")
            return data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    if use_synthetic:
        print(
            "Generating synthetic training data (--allow-synthetic). "
            "Do not publish these metrics as MIT-BIH results."
        )
        beat_data_dir.mkdir(parents=True, exist_ok=True)
        return generate_synthetic_dataset(
            n_samples_per_class=200, beat_length=256, test_ratio=0.2
        )
    raise FileNotFoundError(
        f"No beat data under {beat_data_dir} and synthetic generation disabled. "
        "Pass --allow-synthetic for offline smoke tests only."
    )


def train_cnn(
    beat_data_file: Optional[str] = None,
    beat_data_dir: str = "data/processed",
    output_dir: str = "models",
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_synthetic: bool = False,
    seed: int = 42,
) -> Dict:
    if not TORCH_AVAILABLE:
        return {"status": "torch_not_available"}
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        X_train, y_train, X_test, y_test = load_or_generate_data(
            beat_data_file=beat_data_file,
            beat_data_dir=beat_data_dir,
            use_synthetic=use_synthetic,
        )
    except FileNotFoundError as e:
        return {"status": "data_not_found", "error": str(e)}
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    model = ECG1DCNN(num_classes=5, dropout_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_counts = np.maximum(np.bincount(y_train), 1)
    class_weights = torch.from_numpy(
        len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(num_epochs):
        model.train()
        for beats, batch_labels in train_loader:
            beats = beats.unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(beats), batch_labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for beats, batch_labels in test_loader:
            preds = torch.argmax(model(beats.unsqueeze(1).to(device)), dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch_labels.numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics = {
        "model": "1D-CNN (PyTorch)",
        "data_source": "synthetic" if use_synthetic else "file",
        "num_epochs": num_epochs,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_labels": ["N", "V", "S", "F", "Q"],
        "disclaimer": "Research/educational only — not a medical device.",
    }
    with open(output_dir / "cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), output_dir / "cnn_model.pth")
    print(f"macro_f1={macro_f1:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN (fail closed unless --allow-synthetic)"
    )
    parser.add_argument("--beat-data-file", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic data when no NPZ is present (smoke tests only)",
    )
    args = parser.parse_args()
    result = train_cnn(
        beat_data_file=args.beat_data_file,
        beat_data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_synthetic=args.allow_synthetic,
        seed=args.seed,
    )
    if result.get("status") in {"data_not_found", "torch_not_available"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
