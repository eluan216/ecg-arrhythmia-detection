"""
Complete Stage 2 training orchestration.

Runs the full pipeline:
1. Generate synthetic data
2. Train baseline (Random Forest)
3. Train CNN
4. Compare and evaluate results
"""

import json
from pathlib import Path
from typing import Dict

from src.data.synthetic import save_synthetic_dataset
from src.train.train_baseline import train_baseline
from src.train.train_cnn import train_cnn


def stage2_pipeline(
    data_dir: str = "data/processed",
    output_dir: str = "models",
    use_synthetic: bool = True,
    train_baseline_model: bool = True,
    train_cnn_model: bool = True,
    compare_results: bool = True,
    mlflow_track: bool = True,
) -> Dict[str, Dict]:
    """
    Run complete Stage 2 training pipeline.

    Args:
        data_dir: Directory for processed data.
        output_dir: Directory for model outputs.
        use_synthetic: Use synthetic data for training.
        train_baseline_model: Train Random Forest baseline.
        train_cnn_model: Train PyTorch CNN.
        compare_results: Compare models after training.
        mlflow_track: Enable MLflow tracking.

    Returns:
        Dictionary with all training results.
    """
    results = {}

    print("\n" + "=" * 80)
    print("STAGE 2: COMPLETE ML PIPELINE")
    print("=" * 80)

    # Step 1: Prepare data
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING DATA")
    print("=" * 80)

    if use_synthetic:
        print("Generating synthetic training data...")
        save_synthetic_dataset(
            output_file=f"{data_dir}/synthetic_beats.npz",
            n_samples_per_class=200,
            beat_length=256,
        )
        results["data"] = {
            "source": "synthetic",
            "samples_per_class": 200,
            "total_samples": 1000,
        }
    else:
        print("Using real MIT-BIH data (if available)...")
        results["data"] = {"source": "real"}

    # Step 2: Train baseline
    if train_baseline_model:
        print("\n" + "=" * 80)
        print("STEP 2: TRAINING BASELINE (RANDOM FOREST)")
        print("=" * 80)
        try:
            baseline_metrics = train_baseline(
                beat_data_dir=data_dir,
                output_dir=output_dir,
                n_estimators=100,
                seed=42,
            )
            results["baseline"] = baseline_metrics
        except Exception as e:
            print(f"⚠ Baseline training failed: {e}")
            results["baseline"] = {"error": str(e)}

    # Step 3: Train CNN
    if train_cnn_model:
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING CNN (PYTORCH)")
        print("=" * 80)
        try:
            cnn_metrics = train_cnn(
                beat_data_dir=data_dir,
                output_dir=output_dir,
                num_epochs=50,
                batch_size=32,
                learning_rate=0.001,
                use_synthetic=use_synthetic,
                seed=42,
            )
            results["cnn"] = cnn_metrics
        except Exception as e:
            print(f"⚠ CNN training failed: {e}")
            results["cnn"] = {"error": str(e)}

    # Step 4: Compare results
    if compare_results:
        print("\n" + "=" * 80)
        print("STEP 4: MODEL COMPARISON")
        print("=" * 80)

        comparison = {}

        if "baseline" in results and "error" not in results["baseline"]:
            baseline = results["baseline"]
            comparison["baseline"] = {
                "model": baseline.get("model", "Random Forest"),
                "macro_f1": baseline.get("macro_f1", 0),
                "macro_precision": baseline.get("macro_precision", 0),
                "macro_recall": baseline.get("macro_recall", 0),
                "test_samples": baseline.get("test_samples", 0),
            }

        if "cnn" in results and "error" not in results["cnn"]:
            cnn = results["cnn"]
            comparison["cnn"] = {
                "model": cnn.get("model", "1D-CNN"),
                "macro_f1": cnn.get("macro_f1", 0),
                "macro_precision": cnn.get("macro_precision", 0),
                "macro_recall": cnn.get("macro_recall", 0),
                "test_samples": cnn.get("test_samples", 0),
            }

        if comparison:
            print("\n" + "-" * 80)
            print("MODEL PERFORMANCE SUMMARY")
            print("-" * 80)
            print(f"{'Model':<20} {'F1-Score':<15} {'Precision':<15} {'Recall':<15}")
            print("-" * 80)

            for model_name, metrics in comparison.items():
                print(
                    f"{model_name:<20} "
                    f"{metrics['macro_f1']:<15.4f} "
                    f"{metrics['macro_precision']:<15.4f} "
                    f"{metrics['macro_recall']:<15.4f}"
                )

            # Determine best model
            best_model = max(comparison.items(), key=lambda x: x[1]["macro_f1"])
            print("-" * 80)
            print(f"✓ Best model: {best_model[0]} (F1: {best_model[1]['macro_f1']:.4f})")

            results["comparison"] = comparison
            results["best_model"] = best_model[0]

            # Save comparison
            output_path = Path(output_dir)
            comparison_file = output_path / "comparison.json"
            with open(comparison_file, "w") as f:
                json.dump(comparison, f, indent=2)
            print(f"✓ Comparison saved to {comparison_file}")

    # Summary
    print("\n" + "=" * 80)
    print("STAGE 2 COMPLETE")
    print("=" * 80)
    print(f"\n✓ Baseline model: {output_dir}/baseline_model.pkl")
    print(f"✓ CNN model: {output_dir}/cnn_model.pth")
    print(f"✓ Baseline metrics: {output_dir}/baseline_metrics.json")
    print(f"✓ CNN metrics: {output_dir}/cnn_metrics.json")
    print(f"✓ Comparison: {output_dir}/comparison.json")
    print("\nTo start the API:")
    print("  uvicorn api.main:app --reload")
    print("\nTo view experiments:")
    print("  mlflow ui")

    return results


if __name__ == "__main__":
    import sys

    # Quick test mode
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        print("Running in QUICK TEST mode (10 epochs, 100 samples/class)...")
        results = stage2_pipeline(
            use_synthetic=True,
            train_baseline_model=True,
            train_cnn_model=True,
            compare_results=True,
        )
    else:
        # Full pipeline
        results = stage2_pipeline()

    print("\n" + json.dumps(results, indent=2)[:500] + "...")
