"""Inter-patient (subject-disjoint) evaluation for MIT-BIH arrhythmia classification.

Default protocol: GroupKFold / fixed inter-patient split with subject groups
(records 201/202 share a subject). Reports per-class metrics and confusion matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from src.data.segment import (
    AAMI_CLASSES,
    FEATURE_DIM,
    FEATURE_VERSION,
    create_inter_patient_split,
    load_mitbih_record,
    subject_id_for_record,
)

DISCLAIMER = (
    "Research / educational results only. Not a medical device. "
    "Do not use for clinical decisions."
)


def _metrics_block(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = AAMI_CLASSES
) -> Dict[str, Any]:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_class = {
        c: {
            "precision": float(report[c]["precision"]),
            "recall": float(report[c]["recall"]),
            "f1": float(report[c]["f1-score"]),
            "support": int(report[c]["support"]),
        }
        for c in labels
        if c in report
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def _top_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = AAMI_CLASSES, k: int = 5
) -> List[Dict[str, Any]]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: List[Tuple[str, str, int]] = []
    for i, true_c in enumerate(labels):
        for j, pred_c in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((true_c, pred_c, int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return [
        {"true": t, "pred": p, "count": c} for t, p, c in pairs[:k]
    ]


def load_patient_beats(
    data_dir: Path,
    records: Optional[List[str]] = None,
    max_beats_per_record: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load beats with subject group IDs for GroupKFold.

    Returns X, y, groups, record_ids.
    """
    if records is None:
        # Common MIT-BIH subset used in inter-patient protocols
        records = [
            "100",
            "101",
            "103",
            "105",
            "106",
            "108",
            "109",
            "111",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "118",
            "119",
            "121",
            "122",
            "123",
            "124",
            "200",
            "201",
            "202",
            "203",
            "205",
            "207",
            "208",
            "209",
            "210",
            "212",
            "213",
            "214",
            "215",
            "217",
            "219",
            "220",
            "221",
            "222",
            "223",
            "228",
            "230",
            "231",
            "232",
            "233",
            "234",
        ]

    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    groups_list: List[str] = []
    record_ids: List[str] = []

    for rec in records:
        try:
            beats, labels, meta = load_mitbih_record(
                data_dir, rec, max_beats=max_beats_per_record
            )
        except Exception as exc:
            print(f"Skip record {rec}: {exc}")
            continue
        if len(beats) == 0:
            continue
        subj = subject_id_for_record(rec)
        for beat, lab in zip(beats, labels):
            X_list.append(beat)
            y_list.append(lab)
            groups_list.append(subj)
            record_ids.append(rec)

    if not X_list:
        raise RuntimeError("No beats loaded — check data_dir and record list")

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    return X, y, groups, record_ids


def save_patient_split(
    output_dir: Path,
    train_subjects: List[str],
    test_subjects: List[str],
    train_records: List[str],
    test_records: List[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "protocol": "inter-patient",
        "feature_version": FEATURE_VERSION,
        "train_subjects": sorted(set(train_subjects)),
        "test_subjects": sorted(set(test_subjects)),
        "train_records": sorted(set(train_records)),
        "test_records": sorted(set(test_records)),
    }
    path = output_dir / "patient_split.json"
    path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return path


def run_inter_patient_rf(
    data_dir: Path,
    output_dir: Path,
    n_splits: int = 5,
    max_beats_per_record: Optional[int] = None,
    allow_synthetic: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train/evaluate RF under subject-disjoint protocol."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer real MIT-BIH; synthetic only if explicitly allowed
    try:
        X, y, groups, record_ids = load_patient_beats(
            data_dir, max_beats_per_record=max_beats_per_record
        )
        dataset_type = "mitbih"
    except Exception as exc:
        if not allow_synthetic:
            raise RuntimeError(
                f"Failed to load MIT-BIH from {data_dir}: {exc}. "
                "Pass --allow-synthetic only for offline smoke tests."
            ) from exc
        print(f"MIT-BIH unavailable ({exc}); using synthetic fallback")
        rng = np.random.default_rng(seed)
        n = 500
        X = rng.normal(size=(n, FEATURE_DIM)).astype(np.float64)
        y = rng.choice(AAMI_CLASSES, size=n)
        groups = np.array([f"s{i % 10}" for i in range(n)])
        record_ids = [f"r{i % 20}" for i in range(n)]
        dataset_type = "synthetic"

    # Fixed inter-patient split for held-out reporting
    train_idx, test_idx, train_recs, test_recs = create_inter_patient_split(
        list(dict.fromkeys(record_ids))
    )
    # Map records -> indices
    rec_to_indices: Dict[str, List[int]] = {}
    for i, r in enumerate(record_ids):
        rec_to_indices.setdefault(r, []).append(i)

    train_indices = [i for r in train_recs for i in rec_to_indices.get(r, [])]
    test_indices = [i for r in test_recs for i in rec_to_indices.get(r, [])]

    if not train_indices or not test_indices:
        # Fall back to GroupKFold first fold if fixed split empty (synthetic)
        gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        train_indices, test_indices = next(gkf.split(X, y, groups))

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    # Assert subject-disjoint
    train_subj = set(groups_train.tolist())
    test_subj = set(groups_test.tolist())
    overlap = train_subj & test_subj
    if overlap:
        raise AssertionError(f"Subject leakage detected: {overlap}")

    save_patient_split(
        output_dir,
        train_subjects=list(train_subj),
        test_subjects=list(test_subj),
        train_records=list({record_ids[i] for i in train_indices}),
        test_records=list({record_ids[i] for i in test_indices}),
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    }

    results: Dict[str, Any] = {
        "protocol": "inter-patient",
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "dataset_type": dataset_type,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_train_subjects": len(train_subj),
        "n_test_subjects": len(test_subj),
        "disclaimer": DISCLAIMER,
        "models": {},
    }

    # GroupKFold CV on train subjects
    n_unique = len(np.unique(groups_train))
    cv_splits = min(n_splits, n_unique) if n_unique >= 2 else 0
    for name, clf in models.items():
        cv_scores: List[float] = []
        if cv_splits >= 2:
            gkf = GroupKFold(n_splits=cv_splits)
            for tr, va in gkf.split(X_train, y_train, groups_train):
                clf.fit(X_train[tr], y_train[tr])
                pred = clf.predict(X_train[va])
                cv_scores.append(
                    float(f1_score(y_train[va], pred, average="macro", zero_division=0))
                )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        block = _metrics_block(y_test, y_pred)
        block["group_cv_macro_f1_mean"] = float(np.mean(cv_scores)) if cv_scores else None
        block["group_cv_macro_f1_std"] = float(np.std(cv_scores)) if cv_scores else None
        block["top_confusions"] = _top_confusions(y_test, y_pred)
        results["models"][name] = block

    # Persist primary metrics + model
    best_name = "Random Forest"
    best = results["models"][best_name]
    compat = {
        "protocol": "inter-patient",
        "feature_version": FEATURE_VERSION,
        "dataset_type": dataset_type,
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

    try:
        import joblib

        joblib.dump(models["Random Forest"], output_dir / "baseline_model.pkl")
        meta = {
            "feature_version": FEATURE_VERSION,
            "feature_dim": FEATURE_DIM,
            "dataset_type": dataset_type,
            "protocol": "inter-patient",
            "classes": AAMI_CLASSES,
        }
        (output_dir / "baseline_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover
        print(f"Model persist skipped: {exc}")

    full_path = output_dir / "inter_patient_results.json"
    full_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(compat, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Inter-patient MIT-BIH evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mitbih"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-beats-per-record", type=int, default=None)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow synthetic fallback when MIT-BIH is unavailable (smoke tests only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_inter_patient_rf(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        max_beats_per_record=args.max_beats_per_record,
        allow_synthetic=args.allow_synthetic,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
