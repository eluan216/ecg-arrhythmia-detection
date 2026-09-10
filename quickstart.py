#!/usr/bin/env python
"""
Quick setup and training script for the ECG Arrhythmia Detection project.

This script:
1. Verifies the environment
2. Generates synthetic training data
3. Trains both baseline and CNN models
4. Compares results
5. Validates the API can load models
"""

import json
import sys
from pathlib import Path


def main():
    """Run complete setup and training pipeline."""
    print("\n" + "=" * 80)
    print("ECG ARRHYTHMIA DETECTION - QUICK START")
    print("=" * 80)

    # Check dependencies
    print("\n[1/5] Checking dependencies...")
    try:
        import numpy
        import torch
        import sklearn
        import fastapi
        import mlflow

        print("✓ All dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -e '.[dev]'")
        return 1

    # Create directories
    print("\n[2/5] Creating directories...")
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")

    # Generate synthetic data
    print("\n[3/5] Generating synthetic training data...")
    try:
        from src.data.synthetic import save_synthetic_dataset

        save_synthetic_dataset(
            output_file="data/processed/synthetic_beats.npz",
            n_samples_per_class=200,
            beat_length=256,
        )
        print("✓ Synthetic data generated")
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return 1

    # Train models
    print("\n[4/5] Training models (this may take 1-2 minutes)...")
    try:
        from src.train.train_baseline import train_baseline
        from src.train.train_cnn import train_cnn

        print("\n  Training Random Forest baseline...")
        baseline_metrics = train_baseline(
            beat_data_dir="data/processed",
            output_dir="models",
        )
        print(f"  ✓ Baseline F1-Score: {baseline_metrics.get('macro_f1', 0):.4f}")

        print("\n  Training 1D-CNN...")
        cnn_metrics = train_cnn(
            beat_data_dir="data/processed",
            output_dir="models",
            num_epochs=50,
            use_synthetic=True,
        )
        print(f"  ✓ CNN F1-Score: {cnn_metrics.get('macro_f1', 0):.4f}")

        # Compare
        print("\n" + "-" * 80)
        print("MODEL COMPARISON")
        print("-" * 80)
        print(
            f"{'Model':<20} {'F1-Score':<15} {'Precision':<15} {'Recall':<15}"
        )
        print("-" * 80)
        print(
            f"{'Random Forest':<20} "
            f"{baseline_metrics.get('macro_f1', 0):<15.4f} "
            f"{baseline_metrics.get('macro_precision', 0):<15.4f} "
            f"{baseline_metrics.get('macro_recall', 0):<15.4f}"
        )
        print(
            f"{'1D-CNN':<20} "
            f"{cnn_metrics.get('macro_f1', 0):<15.4f} "
            f"{cnn_metrics.get('macro_precision', 0):<15.4f} "
            f"{cnn_metrics.get('macro_recall', 0):<15.4f}"
        )
        print("-" * 80)

        # Save comparison
        comparison = {
            "baseline": {
                "f1": baseline_metrics.get("macro_f1", 0),
                "precision": baseline_metrics.get("macro_precision", 0),
                "recall": baseline_metrics.get("macro_recall", 0),
            },
            "cnn": {
                "f1": cnn_metrics.get("macro_f1", 0),
                "precision": cnn_metrics.get("macro_precision", 0),
                "recall": cnn_metrics.get("macro_recall", 0),
            },
        }
        with open("models/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Verify API setup
    print("\n[5/5] Verifying API setup...")
    try:
        baseline_file = Path("models/baseline_model.pkl")
        cnn_file = Path("models/cnn_model.pth")

        if baseline_file.exists():
            print(f"✓ Baseline model: {baseline_file}")
        if cnn_file.exists():
            print(f"✓ CNN model: {cnn_file}")

        if baseline_file.exists() or cnn_file.exists():
            print("✓ Models ready for API inference")
        else:
            print("✗ No trained models found")
            return 1

    except Exception as e:
        print(f"✗ API verification failed: {e}")
        return 1

    # Summary
    print("\n" + "=" * 80)
    print("✓ SETUP COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("\n1. Start the API:")
    print("   uvicorn api.main:app --reload")
    print("\n2. View interactive docs:")
    print("   http://localhost:8000/docs")
    print("\n3. View experiment tracking:")
    print("   mlflow ui")
    print("\n4. Run tests:")
    print("   pytest tests/ -v")
    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
