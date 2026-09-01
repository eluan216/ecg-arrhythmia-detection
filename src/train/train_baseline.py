"""Train scikit-learn Random Forest baseline on ECG data."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.data.download import MITBIH_RECORDS_FILTERED
from src.data.segment import (
    AAMI_CLASSES,
    create_inter_patient_split,
    extract_morphology_features,
    load_mitbih_record,
)
from src.models.baseline_sklearn import ECGBaselineRF


def load_and_preprocess_dataset(
    record_ids: List[int],
    record_dir: str = "data/raw",
    beat_length: int = 250,
    extract_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple MIT-BIH records and extract features.

    Args:
        record_ids: List of record IDs to load.
        record_dir: Directory containing MIT-BIH records.
        beat_length: Length of beat window in samples.
        extract_features: If True, extract handcrafted features;
                         if False, return raw beat windows.

    Returns:
        Tuple of (features/beats, labels) where both are numpy arrays.
    """
    all_data = []
    all_labels = []

    print(f"Loading {len(record_ids)} records...")

    for i, record_id in enumerate(record_ids):
        try:
            beat_windows, labels, dist_info = load_mitbih_record(
                record_id,
                record_dir=record_dir,
                beat_length=beat_length,
                channel=0,  # Use first lead (MLII)
            )

            if extract_features:
                # Extract handcrafted features
                features = np.array([extract_morphology_features(beat) for beat in beat_windows])
                all_data.append(features)
            else:
                all_data.append(beat_windows)

            all_labels.append(labels)

            if (i + 1) % max(1, len(record_ids) // 10) == 0:
                print(f"  Loaded {i+1}/{len(record_ids)} records")
                if dist_info:
                    print(f"    Distribution: {dist_info[0]}")

        except Exception as e:
            print(f"  Warning: Failed to load record {record_id}: {e}")
            continue

    if not all_data:
        raise RuntimeError("Failed to load any records from the dataset")

    X = np.vstack(all_data)
    y = np.hstack(all_labels)

    print(f"Total beats loaded: {len(y)}")

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = AAMI_CLASSES[cls_idx]
        pct = 100 * count / len(y)
        print(f"  {cls_name}: {count:6d} ({pct:5.1f}%)")

    return X, y


def train_baseline(
    beat_data_dir: str = "data/raw",
    output_dir: str = "models",
    n_estimators: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Train and evaluate the scikit-learn Random Forest baseline model.

    Uses inter-patient train/test split for rigorous evaluation.

    Args:
        beat_data_dir: Directory containing MIT-BIH records.
        output_dir: Directory to save trained model and metrics.
        n_estimators: Number of trees in Random Forest.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with training and evaluation metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1: TRAINING SCIKIT-LEARN RANDOM FOREST BASELINE")
    print("=" * 80)

    # Create inter-patient split
    train_records, test_records = create_inter_patient_split(
        MITBIH_RECORDS_FILTERED,
        train_ratio=0.8,
        seed=seed,
    )

    print("\nTrain/Test split (by patient):")
    print(f"  Training patients: {len(train_records)}")
    print(f"  Test patients:     {len(test_records)}")

    # Load and preprocess training data
    print(f"\nLoading training data from {beat_data_dir}/")
    X_train, y_train = load_and_preprocess_dataset(
        train_records,
        record_dir=beat_data_dir,
        extract_features=True,
    )

    # Load and preprocess test data
    print(f"\nLoading test data from {beat_data_dir}/")
    X_test, y_test = load_and_preprocess_dataset(
        test_records,
        record_dir=beat_data_dir,
        extract_features=True,
    )

    # Check for data imbalance and apply class weighting
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nClass imbalance observed. Using balanced class weights in Random Forest.")

    # Train model
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    model = ECGBaselineRF(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    # Compute metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    cm = confusion_matrix(y_test, y_pred)

    # Detailed report
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print("Macro-averaged metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")

    print("\nPer-class metrics:")
    print(f"  {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("  " + "-" * 44)
    for i, cls in enumerate(AAMI_CLASSES):
        print(f"  {cls:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=AAMI_CLASSES))

    # Organize metrics dictionary
    metrics = {
        "model": "Random Forest Baseline",
        "n_estimators": n_estimators,
        "feature_type": "handcrafted morphology",
        "n_features": X_train.shape[1],
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall],
            "f1": [float(f) for f in f1],
            "support": [int(s) for s in support],
        },
        "confusion_matrix": cm.tolist(),
        "class_labels": AAMI_CLASSES,
    }

    # Log to MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("ecg-baseline")
        with mlflow.start_run(run_name="random_forest"):
            mlflow.log_params(
                {
                    "model": "Random Forest",
                    "n_estimators": n_estimators,
                    "train_records": len(train_records),
                    "test_records": len(test_records),
                }
            )

            mlflow.log_metrics(
                {
                    "macro_precision": macro_precision,
                    "macro_recall": macro_recall,
                    "macro_f1": macro_f1,
                    "train_samples": len(y_train),
                    "test_samples": len(y_test),
                }
            )

            for i, cls in enumerate(AAMI_CLASSES):
                mlflow.log_metrics(
                    {
                        f"precision_{cls}": float(precision[i]),
                        f"recall_{cls}": float(recall[i]),
                        f"f1_{cls}": float(f1[i]),
                        f"support_{cls}": int(support[i]),
                    }
                )

            mlflow.sklearn.log_model(model.model, "random_forest_model")
            print("\n✓ Metrics logged to MLflow")

    # Save metrics to JSON
    metrics_file = output_dir / "baseline_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_file}")

    # Save model
    import joblib

    model_file = output_dir / "baseline_model.pkl"
    joblib.dump(model.model, model_file)
    print(f"✓ Model saved to {model_file}")

    print("\n" + "=" * 80)
    print("STAGE 1 BASELINE TRAINING COMPLETE")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    train_baseline()
