"""Train PyTorch 1D-CNN on ECG beats.

Default is fail-closed: real beat data required unless --allow-synthetic is set.

To measure inter-patient MIT-BIH numbers (same protocol as RF):
  python -m src.train.train_cnn --from-patient-split reports/mitbih/2026-11-05/patient_split.json

That path loads raw 250-sample windows from data/raw using the frozen subject-disjoint
split and writes metrics that can be published alongside the RF table.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

if TORCH_AVAILABLE:
    from src.data.synthetic import generate_synthetic_dataset
    from src.models.cnn_pytorch import ECG1DCNN


AAMI_CLASSES = ["N", "V", "S", "F", "Q"]


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
            n_samples_per_class=200, beat_length=250, test_ratio=0.2
        )
    raise FileNotFoundError(
        f"No beat data under {beat_data_dir} and synthetic generation disabled. "
        "Pass --allow-synthetic for offline smoke tests only, or "
        "--from-patient-split for the real inter-patient MIT-BIH path."
    )


def load_raw_beats_from_split(
    patient_split_path: str | Path,
    data_dir: str | Path = "data/raw",
    beat_length: int = 250,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load raw beat windows using a frozen patient_split.json (same as RF protocol)."""
    from src.data.segment import load_mitbih_record, subject_id_for_record

    split = json.loads(Path(patient_split_path).read_text(encoding="utf-8"))
    train_records = [int(r) for r in split["train_records"]]
    test_records = [int(r) for r in split["test_records"]]
    data_dir = Path(data_dir)

    def _load(records: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        windows: List[np.ndarray] = []
        labels: List[int] = []
        for rec in records:
            beats, labs, _ = load_mitbih_record(
                rec, record_dir=str(data_dir), beat_length=beat_length, channel=channel
            )
            windows.append(beats.astype(np.float32))
            labels.append(labs.astype(np.int64))
        if not windows:
            raise RuntimeError(f"No beats loaded from records {records}")
        return np.concatenate(windows, axis=0), np.concatenate(labels, axis=0)

    X_train, y_train = _load(train_records)
    X_test, y_test = _load(test_records)

    # Subject-disjoint sanity check
    train_subj = {subject_id_for_record(r) for r in train_records}
    test_subj = {subject_id_for_record(r) for r in test_records}
    if train_subj & test_subj:
        raise AssertionError(f"Subject leakage: {train_subj & test_subj}")

    meta = {
        "protocol": "inter-patient",
        "patient_split": str(patient_split_path),
        "train_records": train_records,
        "test_records": test_records,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_subjects": len(train_subj),
        "n_test_subjects": len(test_subj),
        "beat_length": beat_length,
        "channel": channel,
        "dataset_type": "mitbih",
    }
    return X_train, y_train, X_test, y_test, meta


def train_cnn(
    beat_data_file: Optional[str] = None,
    beat_data_dir: str = "data/processed",
    output_dir: str = "models",
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    use_synthetic: bool = False,
    seed: int = 42,
    patient_split: Optional[str] = None,
    data_raw_dir: str = "data/raw",
) -> Dict:
    if not TORCH_AVAILABLE:
        return {"status": "torch_not_available"}
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    meta: Dict[str, Any] = {}
    if patient_split:
        print(f"Loading raw beats from inter-patient split: {patient_split}", flush=True)
        X_train, y_train, X_test, y_test, meta = load_raw_beats_from_split(
            patient_split, data_dir=data_raw_dir
        )
        data_source = "mitbih_inter_patient"
    else:
        try:
            X_train, y_train, X_test, y_test = load_or_generate_data(
                beat_data_file=beat_data_file,
                beat_data_dir=beat_data_dir,
                use_synthetic=use_synthetic,
            )
            data_source = "synthetic" if use_synthetic else "file"
        except FileNotFoundError as e:
            return {"status": "data_not_found", "error": str(e)}

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # Per-beat z-score (helps CNN on raw amplitude variation)
    def _zscore(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return (X - mu) / sd

    X_train = _zscore(X_train)
    X_test = _zscore(X_test)

    print(
        f"Train: {X_train.shape}, Test: {X_test.shape}, "
        f"classes train={np.bincount(y_train, minlength=5).tolist()}",
        flush=True,
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = ECG1DCNN(num_classes=5, dropout_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_counts = np.maximum(np.bincount(y_train, minlength=5), 1)
    class_weights = torch.from_numpy(
        (len(y_train) / (5.0 * class_counts.astype(np.float32)))
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro = -1.0
    best_state = None
    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for beats, batch_labels in train_loader:
            beats = beats.unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(beats), batch_labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # quick holdout check
            model.eval()
            y_pred_e, y_true_e = [], []
            with torch.no_grad():
                for beats, batch_labels in test_loader:
                    preds = torch.argmax(model(beats.unsqueeze(1).to(device)), dim=1)
                    y_pred_e.extend(preds.cpu().numpy())
                    y_true_e.extend(batch_labels.numpy())
            mf1 = float(
                f1_score(y_true_e, y_pred_e, average="macro", zero_division=0)
            )
            print(
                f"Epoch {epoch+1}/{num_epochs}  loss={running/max(n_batches,1):.4f}  "
                f"holdout_macro_f1={mf1:.4f}",
                flush=True,
            )
            if mf1 > best_macro:
                best_macro = mf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

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
        y_true, y_pred, labels=list(range(5)), average=None, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    per_class = {
        AAMI_CLASSES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(5)
    }

    metrics: Dict[str, Any] = {
        "model": "1D-CNN (PyTorch)",
        "data_source": data_source,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
        "class_labels": AAMI_CLASSES,
        "disclaimer": "Research/educational only — not a medical device.",
        **meta,
    }

    out_metrics = output_dir / "cnn_metrics.json"
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), output_dir / "cnn_model.pth")
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    print(f"Wrote {out_metrics}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN (fail closed unless --allow-synthetic)"
    )
    parser.add_argument("--beat-data-file", default=None)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic data when no NPZ is present (smoke tests only)",
    )
    parser.add_argument(
        "--from-patient-split",
        default=None,
        help="Path to patient_split.json (inter-patient MIT-BIH protocol)",
    )
    parser.add_argument(
        "--data-raw-dir",
        default="data/raw",
        help="Directory containing MIT-BIH .dat/.hea/.atr (used with --from-patient-split)",
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
        patient_split=args.from_patient_split,
        data_raw_dir=args.data_raw_dir,
    )
    if result.get("status") in ("data_not_found", "torch_not_available"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
