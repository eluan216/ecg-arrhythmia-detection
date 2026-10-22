"""
Inter-patient evaluation protocol (default rigor path).

No beat from the same patient appears in both train and test.
Patient IDs, seeds, and metrics are written to disk for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.download import MITBIH_RECORDS_FILTERED, EXCLUDED_RECORDS
from src.data.segment import (
    AAMI_CLASSES,
    create_inter_patient_split,
    extract_morphology_features,
    load_mitbih_record,
)

SEED = 42
DEFAULT_TRAIN_RATIO = 0.8


def save_patient_split(
    train_records: List[int],
    test_records: List[int],
    path: str | Path = "models/patient_split.json",
    seed: int = SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "excluded_records": EXCLUDED_RECORDS,
        "exclusion_reason": "Paced-beat records commonly excluded in MIT-BIH literature",
        "train_patient_ids": sorted(train_records),
        "test_patient_ids": sorted(test_records),
        "n_train_patients": len(train_records),
        "n_test_patients": len(test_records),
        "protocol": "inter-patient: no beat from the same patient in both train and test",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_patient_beats(
    record_ids: List[int],
    record_dir: str = "data/raw",
    beat_length: int = 250,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X features, y labels, and patient-id group vector."""
    feats, labels, groups = [], [], []
    for rid in record_ids:
        try:
            windows, y, _ = load_mitbih_record(
                rid, record_dir=record_dir, beat_length=beat_length, channel=0
            )
        except Exception as e:
            print(f"  skip record {rid}: {e}")
            continue
        for beat, lab in zip(windows, y):
            feats.append(extract_morphology_features(beat))
            labels.append(int(lab))
            groups.append(int(rid))
    if not feats:
        raise RuntimeError(
            f"No beats loaded from {record_dir}. Run: python -m src.data.download"
        )
    return np.asarray(feats), np.asarray(labels), np.asarray(groups)


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(AAMI_CLASSES))), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            "labels": AAMI_CLASSES,
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "f1": [float(x) for x in f1],
            "support": [int(x) for x in support],
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(AAMI_CLASSES)))
        ).tolist(),
    }


def group_cv_macro_f1(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> Tuple[float, float, List[float]]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        return float("nan"), float("nan"), []
    gkf = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(
        model, X, y, cv=gkf, groups=groups, scoring="f1_macro", n_jobs=-1
    )
    return float(scores.mean()), float(scores.std()), [float(s) for s in scores]


def run_inter_patient_rf(
    record_dir: str = "data/raw",
    output_dir: str = "models",
    seed: int = SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_estimators: int = 150,
    include_logistic: bool = True,
) -> Dict[str, Any]:
    """
    Full inter-patient RF (+ optional LR) evaluation on MIT-BIH.

    Writes:
      models/patient_split.json
      models/inter_patient_results.json
      models/baseline_metrics.json (compat with existing evaluate.py)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records, test_records = create_inter_patient_split(
        MITBIH_RECORDS_FILTERED, train_ratio=train_ratio, seed=seed
    )
    split_path = save_patient_split(
        train_records, test_records, output_dir / "patient_split.json", seed, train_ratio
    )

    print("Inter-patient protocol")
    print(f"  Train patients ({len(train_records)}): {sorted(train_records)}")
    print(f"  Test patients  ({len(test_records)}): {sorted(test_records)}")
    print(f"  Split file → {split_path}")

    print("Loading train beats...")
    X_train, y_train, g_train = load_patient_beats(train_records, record_dir)
    print("Loading test beats...")
    X_test, y_test, g_test = load_patient_beats(test_records, record_dir)

    assert set(g_train).isdisjoint(set(g_test)), "Patient leakage detected"

    results: Dict[str, Any] = {
        "protocol": "inter-patient",
        "seed": seed,
        "train_ratio": train_ratio,
        "excluded_records": EXCLUDED_RECORDS,
        "train_patient_ids": sorted(train_records),
        "test_patient_ids": sorted(test_records),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "feature_type": "handcrafted morphology (15-D)",
        "models": {},
        "disclaimer": "Research/educational only — not a medical device.",
    }

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=16,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if include_logistic:
        models["Logistic Regression"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        block = _metrics_block(y_test, y_pred)

        # Group-aware CV on training patients only
        cv_model = models[name].__class__(**models[name].get_params()) if hasattr(
            models[name], "get_params"
        ) else model
        try:
            # Rebuild fresh estimator for CV
            if name == "Random Forest":
                cv_est = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=16,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                )
            else:
                cv_est = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                )
            cv_mean, cv_std, cv_scores = group_cv_macro_f1(
                cv_est, X_train, y_train, g_train, n_splits=5
            )
        except Exception as e:
            print(f"  Group CV skipped: {e}")
            cv_mean, cv_std, cv_scores = float("nan"), float("nan"), []

        block["group_cv_macro_f1_mean"] = cv_mean
        block["group_cv_macro_f1_std"] = cv_std
        block["group_cv_scores"] = cv_scores
        results["models"][name] = block

        print(
            f"  holdout macro-F1={block['macro_f1']:.4f}  "
            f"weighted-F1={block['weighted_f1']:.4f}  "
            f"acc={block['accuracy']:.4f}"
        )
        if cv_scores:
            print(f"  GroupKFold macro-F1={cv_mean:.4f} ± {cv_std:.4f}")

        # Error analysis notes
        cm = np.asarray(block["confusion_matrix"])
        confusions = []
        for i, row in enumerate(cm):
            for j, v in enumerate(row):
                if i != j and v > 0:
                    confusions.append(
                        {
                            "true": AAMI_CLASSES[i],
                            "pred": AAMI_CLASSES[j],
                            "count": int(v),
                        }
                    )
        confusions.sort(key=lambda x: -x["count"])
        block["top_confusions"] = confusions[:10]

    # Best by macro-F1
    best_name = max(results["models"], key=lambda n: results["models"][n]["macro_f1"])
    results["selected_model"] = best_name
    results["selection_rule"] = "highest macro-F1 on inter-patient holdout"

    out_path = output_dir / "inter_patient_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")

    # Compat metrics for evaluate.py / API
    best = results["models"][best_name]
    compat = {
        "model": best_name,
        "data_source": "mitbih",
        "protocol": "inter-patient",
        "seed": seed,
        "train_patient_ids": results["train_patient_ids"],
        "test_patient_ids": results["test_patient_ids"],
        "train_samples": results["train_samples"],
        "test_samples": results["test_samples"],
        "macro_precision": best["macro_precision"],
        "macro_recall": best["macro_recall"],
        "macro_f1": best["macro_f1"],
        "weighted_f1": best["weighted_f1"],
        "accuracy": best["accuracy"],
        "per_class": best["per_class"],
        "confusion_matrix": best["confusion_matrix"],
        "group_cv_macro_f1_mean": best.get("group_cv_macro_f1_mean"),
        "group_cv_macro_f1_std": best.get("group_cv_macro_f1_std"),
        "top_confusions": best.get("top_confusions", []),
        "class_labels": AAMI_CLASSES,
        "disclaimer": results["disclaimer"],
    }
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(compat, indent=2), encoding="utf-8"
    )

    # Persist RF model for API
    try:
        import joblib

        rf = models["Random Forest"]
        joblib.dump(rf, output_dir / "baseline_model.pkl")
    except Exception as e:
        print(f"Model pickle skipped: {e}")

    return results


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Inter-patient MIT-BIH evaluation")
    p.add_argument("--record-dir", default="data/raw")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run_inter_patient_rf(
        record_dir=args.record_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
