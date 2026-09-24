"""Train scikit-learn Random Forest baseline on ECG data.

Uses real MIT-BIH records when available under data/raw.
Falls back to synthetic morphology features if no records load
(so Colab / offline demos never hard-fail with empty data).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

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
    """Load multiple MIT-BIH records and extract features or raw windows."""
    all_data = []
    all_labels = []

    print(f"Loading {len(record_ids)} records from {record_dir}/...")

    for i, record_id in enumerate(record_ids):
        try:
            beat_windows, labels, dist_info = load_mitbih_record(
                record_id,
                record_dir=record_dir,
                beat_length=beat_length,
                channel=0,
            )
            if extract_features:
                features = np.array(
                    [extract_morphology_features(beat) for beat in beat_windows]
                )
                all_data.append(features)
            else:
                all_data.append(beat_windows)
            all_labels.append(labels)
            if (i + 1) % max(1, len(record_ids) // 10) == 0:
                print(f"  Loaded {i + 1}/{len(record_ids)} records")
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
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls_idx, count in zip(unique, counts):
        cls_name = AAMI_CLASSES[cls_idx]
        pct = 100 * count / len(y)
        print(f"  {cls_name}: {count:6d} ({pct:5.1f}%)")
    return X, y


def _synthetic_feature_split(
    n_per_class: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Offline train/test split using synthetic beats + morphology features."""
    from src.data.synthetic import generate_synthetic_dataset

    print(
        "No MIT-BIH records found — using synthetic ECG features "
        "(set use_synthetic=False to require real data)."
    )
    X_train_b, y_train, X_test_b, y_test = generate_synthetic_dataset(
        n_samples_per_class=n_per_class,
        beat_length=250,
        test_ratio=0.2,
        seed=seed,
    )
    X_train = np.array([extract_morphology_features(b) for b in X_train_b])
    X_test = np.array([extract_morphology_features(b) for b in X_test_b])
    return X_train, y_train, X_test, y_test


def train_baseline(
    beat_data_dir: str = "data/raw",
    output_dir: str = "models",
    n_estimators: int = 100,
    seed: int = 42,
    use_synthetic: bool = True,
) -> Dict:
    """
    Train and evaluate Random Forest baseline.

    Prefer inter-patient MIT-BIH data under beat_data_dir.
    If no records load and use_synthetic=True, train on synthetic features.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1: TRAINING SCIKIT-LEARN RANDOM FOREST BASELINE")
    print("=" * 80)

    data_source = "mitbih"
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    train_records: List[int] = []
    test_records: List[int] = []

    try:
        train_records, test_records = create_inter_patient_split(
            MITBIH_RECORDS_FILTERED,
            train_ratio=0.8,
            seed=seed,
        )
        print("\nTrain/Test split (by patient):")
        print(f"  Training patients: {len(train_records)}")
        print(f"  Test patients:     {len(test_records)}")
        print(f"\nLoading training data from {beat_data_dir}/")
        X_train, y_train = load_and_preprocess_dataset(
            train_records,
            record_dir=beat_data_dir,
            extract_features=True,
        )
        print(f"\nLoading test data from {beat_data_dir}/")
        X_test, y_test = load_and_preprocess_dataset(
            test_records,
            record_dir=beat_data_dir,
            extract_features=True,
        )
    except RuntimeError as e:
        if not use_synthetic:
            raise
        print(f"\nMIT-BIH load failed: {e}")
        X_train, y_train, X_test, y_test = _synthetic_feature_split(seed=seed)
        data_source = "synthetic"
        train_records, test_records = [], []

    assert X_train is not None and y_train is not None
    assert X_test is not None and y_test is not None

    print("\nClass imbalance: using balanced class weights in Random Forest.")
    print(f"\nTraining Random Forest with {n_estimators} estimators...")
    model = ECGBaselineRF(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = model.predict(X_test)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Data source: {data_source}")
    print("Macro-averaged metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-Score:  {macro_f1:.4f}")
    print("\nPer-class metrics:")
    print(f"  {'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("  " + "-" * 44)
    for i, cls in enumerate(AAMI_CLASSES):
        if i < len(precision):
            print(f"  {cls:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=AAMI_CLASSES, zero_division=0))

    metrics = {
        "model": "Random Forest Baseline",
        "data_source": data_source,
        "n_estimators": n_estimators,
        "feature_type": "handcrafted morphology",
        "n_features": int(X_train.shape[1]),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall],
            "f1": [float(f) for f in f1],
            "support": [int(s) for s in support],
        },
        "confusion_matrix": cm.tolist(),
        "class_labels": AAMI_CLASSES,
        "disclaimer": "Research/educational only — not a medical device.",
    }

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("ecg-baseline")
            with mlflow.start_run(run_name="random_forest"):
                mlflow.log_params(
                    {
                        "model": "Random Forest",
                        "data_source": data_source,
                        "n_estimators": n_estimators,
                    }
                )
                mlflow.log_metrics(
                    {
                        "macro_precision": macro_precision,
                        "macro_recall": macro_recall,
                        "macro_f1": macro_f1,
                    }
                )
                mlflow.sklearn.log_model(model.model, "random_forest_model")
                print("\n✓ Metrics logged to MLflow")
        except Exception as e:
            print(f"MLflow logging skipped: {e}")

    metrics_file = output_dir / "baseline_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_file}")

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
