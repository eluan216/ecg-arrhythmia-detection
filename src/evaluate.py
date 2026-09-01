"""Evaluation module for comparing models."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_metrics(model_dir: str = "models") -> Dict[str, Dict]:
    """
    Load metrics from saved model files.
    
    Args:
        model_dir: Directory containing metrics.json files.
    
    Returns:
        Dictionary mapping model names to their metrics.
    """
    model_dir = Path(model_dir)
    all_metrics = {}
    
    for metrics_file in sorted(model_dir.glob("*_metrics.json")):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            model_name = metrics.get('model', metrics_file.stem)
            all_metrics[model_name] = metrics
    
    return all_metrics


def print_comparison_table(all_metrics: Dict[str, Dict]) -> None:
    """
    Print a formatted comparison table of model metrics.
    
    Displays macro and per-class metrics for all models.
    
    Args:
        all_metrics: Dictionary of model metrics.
    """
    if not all_metrics:
        print("No metrics found. Train models first.")
        return
    
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE".center(100))
    print("="*100)
    
    # Summary comparison
    print("\nSUMMARY METRICS:")
    print("-" * 100)
    print(f"{'Model':<25} {'Samples':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 100)
    
    for model_name, metrics in all_metrics.items():
        train_samples = metrics.get('train_samples', 0)
        test_samples = metrics.get('test_samples', 0)
        samples_str = f"{train_samples}/{test_samples}"
        
        macro_prec = metrics.get('macro_precision', metrics.get('per_class_precision', [0])[0])
        macro_rec = metrics.get('macro_recall', metrics.get('per_class_recall', [0])[0])
        macro_f1 = metrics.get('macro_f1', 0.0)
        
        print(f"{model_name:<25} {samples_str:<15} {macro_prec:>14.4f} {macro_rec:>14.4f} {macro_f1:>14.4f}")
    
    print("\n" + "="*100)
    print("DETAILED PER-CLASS METRICS")
    print("="*100)
    
    classes = ["N (Normal)", "V (Ventricular)", "S (Supraventricular)", "F (Fusion)", "Q (Paced/Unknown)"]
    
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name}:")
        print("-" * 100)
        
        per_class = metrics.get('per_class', {})
        precision = per_class.get('precision', metrics.get('per_class_precision', []))
        recall = per_class.get('recall', metrics.get('per_class_recall', []))
        f1 = per_class.get('f1', metrics.get('per_class_f1', []))
        support = per_class.get('support', metrics.get('support', []))
        
        print(f"{'Class':<25} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<15}")
        print("-" * 100)
        
        for i, cls_name in enumerate(classes):
            if i < len(precision) and i < len(recall) and i < len(f1):
                supp = support[i] if i < len(support) else 0
                print(
                    f"{cls_name:<25} {precision[i]:>14.4f} {recall[i]:>14.4f} "
                    f"{f1[i]:>14.4f} {supp:>14}"
                )
        
        # Add hyperparameters if available
        if 'num_epochs' in metrics or 'batch_size' in metrics:
            print("\nHyperparameters:")
            for key in ['num_epochs', 'batch_size', 'learning_rate', 'device']:
                if key in metrics:
                    print(f"  {key}: {metrics[key]}")
    
    print("\n" + "="*100)


def save_comparison_json(
    all_metrics: Dict[str, Dict],
    output_file: str = "models/comparison.json",
) -> None:
    """
    Save comparison metrics to JSON file.
    
    Args:
        all_metrics: Dictionary of model metrics.
        output_file: Output JSON file path.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        "summary": {},
        "detailed": all_metrics,
    }
    
    # Build summary
    for model_name, metrics in all_metrics.items():
        comparison["summary"][model_name] = {
            "macro_precision": metrics.get('macro_precision', metrics.get('per_class_precision', [0.0])[0]),
            "macro_recall": metrics.get('macro_recall', metrics.get('per_class_recall', [0.0])[0]),
            "macro_f1": metrics.get('macro_f1', 0.0),
            "train_samples": metrics.get('train_samples', 0),
            "test_samples": metrics.get('test_samples', 0),
        }
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {output_path}")


def compare_models(
    model_dir: str = "models",
    output_file: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Load and compare all available models.
    
    Args:
        model_dir: Directory containing model metrics.
        output_file: Optional file to save comparison.
    
    Returns:
        Dictionary of all metrics.
    """
    metrics = load_metrics(model_dir)
    
    if metrics:
        print_comparison_table(metrics)
        
        if output_file:
            save_comparison_json(metrics, output_file)
    else:
        print(f"No metrics found in {model_dir}")
    
    return metrics


if __name__ == "__main__":
    compare_models()
