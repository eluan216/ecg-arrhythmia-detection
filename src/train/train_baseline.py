"""Train scikit-learn Random Forest baseline on ECG data.

Synthetic morphology features are opt-in via --allow-synthetic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.download import MITBIH_RECORDS_FILTERED
from src.data.segment import (
    AAMI_CLASSES,
    create_inter_patient_split,
    extract_morphology_features,
    load_mitbih_record,
    subject_id_for_record,
)
from src.models.baseline_sklearn import ECGBaselineRF

try:
    from src.eval.inter_patient import save_patient_split
except ImportError:
    save_patient_split = None


def load_and_preprocess_dataset(
    record_ids: List[int],
    record_dir: str = "data/raw",
    beat_length: int = 250,
    extract_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    all_data = []
    all_labels = []
    print(f"Loading {len(record_ids)} records from {record_dir}/...")
    for i, record_id in enumerate(record_ids):
        try:
            beat_windows, labels, dist_info = load_mitbih_record(
                record_id, record_dir=record_dir, beat_length=beat_length, channel=0
            )
            if extract_features:
                features = np.array([extract_morphology_features(beat) for beat in beat_windows])
                all_data.append(features)
            else:
                all_data.append(beat_windows)
            all_labels.append(labels)
        except Exception as e:
            print(f"  Warning: Failed to load record {record_id}: {e}")
            continue
    if not all_data:
        raise RuntimeError("Failed to load any records from the dataset")
    X = np.vstack(all_data)
    y = np.hstack(all_labels)
    print(f"Total beats loaded: {len(y)}")
    return X, y


def _synthetic_feature_split(n_per_class: int = 200, seed: int = 42):
    from src.data.synthetic import generate_synthetic_dataset

    print(
        "Using synthetic ECG features (--allow-synthetic). "
        "Do not publish these metrics as MIT-BIH results."
    )
    X_train_b, y_train, X_test_b, y_test = generate_synthetic_dataset(
        n_samples_per_class=n_per_class, beat_length=250, test_ratio=0.2, seed=seed
    )
    X_train = np.array([extract_morphology_features(b) for b in X_train_b])
    X_test = np.array([extract_morphology_features(b) for b in X_test_b])
    return X_train, y_train, X_test, y_test


def train_baseline(
    beat_data_dir: str = "data/raw",
    output_dir: str = "models",
    n_estimators: int = 100,
    seed: int = 42,
    use_synthetic: bool = False,
) -> Dict:
    """Train RF baseline. Synthetic only when use_synthetic=True (fail closed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("TRAINING SCIKIT-LEARN RANDOM FOREST BASELINE")
    print("=" * 80)
    data_source = "mitbih"
    protocol = "inter-patient"
    train_records: List[int] = []
    test_records: List[int] = []
    try:
        train_records, test_records = create_inter_patient_split(
            MITBIH_RECORDS_FILTERED, train_ratio=0.8, seed=seed
        )
        if save_patient_split is not None:
            save_patient_split(
                output_dir=output_dir,
                train_subjects=[subject_id_for_record(r) for r in train_records],
                test_subjects=[subject_id_for_record(r) for r in test_records],
                train_records=train_records,
                test_records=test_records,
            )
        X_train, y_train = load_and_preprocess_dataset(train_records, record_dir=beat_data_dir)
        X_test, y_test = load_and_preprocess_dataset(test_records, record_dir=beat_data_dir)
    except RuntimeError as e:
        if not use_synthetic:
            raise RuntimeError(
                f"Failed to load MIT-BIH from {beat_data_dir}: {e}. "
                "Pass --allow-synthetic only for offline smoke tests."
            ) from e
        X_train, y_train, X_test, y_test = _synthetic_feature_split(seed=seed)
        data_source = "synthetic"
        protocol = "random-beat (synthetic)"
        train_records, test_records = [], []

    model = ECGBaselineRF(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    metrics = {
        "model": "Random Forest Baseline",
        "data_source": data_source,
        "protocol": protocol,
        "seed": seed,
        "n_estimators": n_estimators,
        "n_features": int(X_train.shape[1]),
        "train_patient_ids": sorted(train_records),
        "test_patient_ids": sorted(test_records),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall],
            "f1": [float(f) for f in f1],
            "support": [int(s) for s in support],
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_labels": AAMI_CLASSES,
        "disclaimer": "Research/educational only — not a medical device.",
    }
    with open(output_dir / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    import joblib

    joblib.dump(model.model, output_dir / "baseline_model.pkl")
    print(f"macro_f1={metrics['macro_f1']:.4f} data_source={data_source}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RF baseline (MIT-BIH by default; synthetic opt-in)"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback when MIT-BIH is unavailable (smoke tests only)",
    )
    args = parser.parse_args()
    train_baseline(
        beat_data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_estimators=args.n_estimators,
        seed=args.seed,
        use_synthetic=args.allow_synthetic,
    )


if __name__ == "__main__":
    main()
