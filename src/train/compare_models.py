"""
Comprehensive Stage 2 pipeline: Train baseline and CNN, then compare.

This script orchestrates the complete Stage 2 workflow:
1. Generate or load synthetic/real ECG data
2. Train scikit-learn Random Forest baseline
3. Train PyTorch 1D-CNN
4. Compare metrics and generate comparison report
5. Log results to MLflow (if available)
"""

import sys
from pathlib import Path
from typing import Dict

# Try importing required modules
try:
    from src.train.train_baseline import train_baseline

    BASELINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import baseline training: {e}")
    BASELINE_AVAILABLE = False

try:
    from src.train.train_cnn import train_cnn

    CNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import CNN training: {e}")
    CNN_AVAILABLE = False

try:
    from src.evaluate import load_metrics

    EVALUATE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import evaluation: {e}")
    EVALUATE_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


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
    Execute complete Stage 2 training and evaluation pipeline.

    Args:
        data_dir: Directory for processed data.
        output_dir: Directory for model outputs.
        use_synthetic: Use synthetic data if real data unavailable.
        train_baseline_model: Train baseline RF model.
        train_cnn_model: Train CNN model.
        compare_results: Compare models after training.
        mlflow_track: Track experiments with MLflow.

    Returns:
        Dictionary of all metrics.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: DEEP LEARNING & MODEL COMPARISON")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    # Train baseline
    if train_baseline_model and BASELINE_AVAILABLE:
        print("\n[1/3] Training Random Forest Baseline...")
        print("-" * 80)
        try:
            baseline_metrics = train_baseline(
                beat_data_dir=data_dir,
                output_dir=output_dir,
            )
            results["baseline"] = baseline_metrics
            print("✓ Baseline training complete")
        except Exception as e:
            print(f"✗ Baseline training failed: {e}")
            results["baseline"] = {"status": "failed", "error": str(e)}

    # Train CNN
    if train_cnn_model and CNN_AVAILABLE:
        print("\n[2/3] Training PyTorch 1D-CNN...")
        print("-" * 80)
        try:
            cnn_metrics = train_cnn(
                beat_data_dir=data_dir,
                output_dir=output_dir,
                use_synthetic=use_synthetic,
                num_epochs=50,
            )
            results["cnn"] = cnn_metrics
            print("✓ CNN training complete")
        except Exception as e:
            print(f"✗ CNN training failed: {e}")
            results["cnn"] = {"status": "failed", "error": str(e)}

    # Compare results
    if compare_results and EVALUATE_AVAILABLE:
        print("\n[3/3] Comparing Models...")
        print("-" * 80)
        try:
            all_metrics = load_metrics(output_dir)
            if all_metrics:
                from src.evaluate import print_comparison_table, save_comparison_json

                print_comparison_table(all_metrics)

                # Save detailed comparison
                comparison_file = Path(output_dir) / "comparison.json"
                save_comparison_json(all_metrics, str(comparison_file))

                results["comparison"] = all_metrics
                print(f"✓ Comparison saved to {comparison_file}")
            else:
                print("No trained models found for comparison")
        except Exception as e:
            print(f"✗ Comparison failed: {e}")

    # Log to MLflow
    if mlflow_track and MLFLOW_AVAILABLE:
        print("\n[MLflow] Logging experiment summary...")
        try:
            mlflow.set_experiment("ecg-stage2-pipeline")
            with mlflow.start_run(run_name="stage2_summary"):
                mlflow.log_dict(results, "stage2_results.json")
                print("✓ Results logged to MLflow")
        except Exception as e:
            print(f"Note: MLflow logging skipped: {e}")

    # Print final summary
    print("\n" + "=" * 80)
    print("STAGE 2 PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("Files generated:")
    print("  - baseline_model.pkl / baseline_metrics.json")
    print("  - cnn_model.pth / cnn_metrics.json")
    print("  - comparison.json")

    return results


def quick_test():
    """Quick test with minimal synthetic data."""
    print("\n" + "=" * 80)
    print("STAGE 2: QUICK TEST WITH SYNTHETIC DATA")
    print("=" * 80)

    data_dir = "data/processed"
    output_dir = "models"

    results = stage2_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        use_synthetic=True,
        train_baseline_model=True,
        train_cnn_model=True,
        compare_results=True,
    )

    return results


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        results = quick_test()
    else:
        results = stage2_pipeline()

    print("\nDone! Review models in the output directory.")
