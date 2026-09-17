"""
Multi-model benchmark pipeline for ECG beat classification.

Demonstrates a repeatable ML engineering pattern:
  synthetic/real features → preprocess → candidate models →
  stratified CV → metric comparison → model selection

Works offline with synthetic data (no PhysioNet download required).

Usage:
    python -m src.pipeline.benchmark
    python -m src.pipeline.benchmark --samples 300
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

AAMI_CLASSES = ["N", "V", "S", "F", "Q"]

# NumPy 2.x renamed trapz → trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: Optional[float]
    cv_f1_mean: float
    cv_f1_std: float
    selected: bool = False


def extract_morphology_features(beats: np.ndarray) -> np.ndarray:
    """Handcrafted morphology features from beat windows."""
    feats = []
    for beat in beats:
        peak = float(np.max(np.abs(beat)))
        mean = float(np.mean(beat))
        std = float(np.std(beat))
        energy = float(np.sum(beat**2))
        peak_idx = int(np.argmax(np.abs(beat)))
        width_proxy = float(np.sum(np.abs(beat) > 0.3 * peak))
        qrs_area = float(_trapz(np.abs(beat)))
        fft = np.abs(np.fft.rfft(beat))
        spectral_centroid = float(np.sum(np.arange(len(fft)) * fft) / (np.sum(fft) + 1e-8))
        feats.append([peak, mean, std, energy, peak_idx, width_proxy, qrs_area, spectral_centroid])
    return np.asarray(feats, dtype=np.float64)


def generate_synthetic_beats(
    n_per_class: int = 250,
    beat_length: int = 250,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight synthetic beats (no external deps beyond numpy)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, beat_length)
    X_list, y_list = [], []

    templates = {
        0: lambda: 2.0 * np.exp(-30 * t**2) + 0.5 * np.exp(-15 * (t - 0.5) ** 2),
        1: lambda: 3.0 * np.exp(-15 * t**2) - 0.3 * np.exp(-10 * (t - 0.6) ** 2),
        2: lambda: 1.5 * np.exp(-40 * (t + 0.3) ** 2) + 0.3 * np.exp(-20 * (t - 0.3) ** 2),
        3: lambda: 2.1 * np.exp(-28 * (t + 0.12) ** 2) - 0.6 * np.exp(-18 * (t - 0.45) ** 2),
        4: lambda: rng.normal(0, 0.5, beat_length),
    }

    for cls, tmpl in templates.items():
        for _ in range(n_per_class):
            beat = tmpl() + 0.5 * np.sin(2 * np.pi * t / 2) + rng.normal(0, 0.1, beat_length)
            X_list.append(beat)
            y_list.append(cls)

    X = np.asarray(X_list)
    y = np.asarray(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def build_candidates(random_state: int = 42) -> Dict[str, Any]:
    """Appropriate models for tabular morphology classification."""
    models: Dict[str, Any] = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def evaluate_on_holdout(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
    }
    try:
        proba = model.predict_proba(X_test)
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["roc_auc_ovr"] = None
    return metrics


def cross_val_f1(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[float, float]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def run_benchmark(
    n_per_class: int = 250,
    output_dir: str = "models",
    selection_metric: str = "f1_macro",
) -> List[ModelResult]:
    """
    Full benchmark:
      synthetic beats → morphology features → candidate models →
      stratified 5-fold CV + holdout metrics → select best by F1-macro
    """
    print("=" * 72)
    print("ECG MODEL BENCHMARK PIPELINE")
    print("=" * 72)
    print("Data → Features → Candidates → Stratified CV → Holdout → Selection")
    print()

    X_beats, y = generate_synthetic_beats(n_per_class=n_per_class)
    X = extract_morphology_features(X_beats)
    print(f"Samples: {len(y)} | Features: {X.shape[1]} | Classes: {len(AAMI_CLASSES)}")
    print(f"Class counts: {dict(zip(AAMI_CLASSES, np.bincount(y)))}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = build_candidates()
    results: List[ModelResult] = []

    print(f"{'Model':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV-F1':>12}")
    print("-" * 72)

    for name, model in candidates.items():
        holdout = evaluate_on_holdout(model, X_train, y_train, X_test, y_test)
        cv_model = build_candidates()[name]
        cv_mean, cv_std = cross_val_f1(cv_model, X_train, y_train)

        auc = holdout.get("roc_auc_ovr")
        row = ModelResult(
            name=name,
            accuracy=holdout["accuracy"],
            precision_macro=holdout["precision_macro"],
            recall_macro=holdout["recall_macro"],
            f1_macro=holdout["f1_macro"],
            roc_auc_ovr=auc,
            cv_f1_mean=cv_mean,
            cv_f1_std=cv_std,
        )
        results.append(row)
        auc_str = f"{auc:.3f}" if auc is not None else "  n/a"
        print(
            f"{name:<22} {row.accuracy:7.3f} {row.precision_macro:7.3f} "
            f"{row.recall_macro:7.3f} {row.f1_macro:7.3f} {auc_str:>7} "
            f"{cv_mean:5.3f}±{cv_std:.3f}"
        )

    best = max(results, key=lambda r: getattr(r, selection_metric))
    for r in results:
        r.selected = r.name == best.name

    print("-" * 72)
    print(f"SELECTED: {best.name}  (criterion: {selection_metric})")
    print(
        "Rationale: macro-F1 rewards balanced performance across AAMI classes "
        "under imbalance; stratified CV guards against split noise."
    )
    print()
    if not HAS_XGB:
        print("Note: install xgboost for XGBoost in the benchmark table.")
    if not HAS_LGBM:
        print("Note: install lightgbm for LightGBM in the benchmark table.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "selection_metric": selection_metric,
        "selected_model": best.name,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "data": "synthetic_morphology_features",
        "disclaimer": "Research/educational only — not a medical device.",
        "results": [asdict(r) for r in results],
    }
    path = out / "benchmark_results.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved → {path}")
    print("=" * 72)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ECG multi-model benchmark pipeline")
    parser.add_argument("--samples", type=int, default=250, help="Samples per AAMI class")
    parser.add_argument("--output-dir", default="models", help="Where to write JSON results")
    args = parser.parse_args()
    run_benchmark(n_per_class=args.samples, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
