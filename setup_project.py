#!/usr/bin/env python
"""
Setup and validation script for ECG Arrhythmia Detection project.

Run this script to:
1. Generate synthetic test data
2. Run test suite
3. Validate project structure
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Verify Python version is 3.9+."""
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9+ required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} OK")


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'pytest',
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg} installed")
        except ImportError:
            missing.append(pkg)
            print(f"✗ {pkg} NOT found")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -e '.[dev]'")
        return False
    
    return True


def generate_synthetic_data():
    """Generate synthetic test dataset."""
    try:
        from src.data.synthetic import save_synthetic_dataset
        
        print("\nGenerating synthetic test dataset...")
        save_synthetic_dataset(
            output_file="data/processed/synthetic_beats.npz",
            n_samples_per_class=200,
            beat_length=256,
        )
        print("✓ Synthetic data generated")
        return True
    except Exception as e:
        print(f"✗ Failed to generate synthetic data: {e}")
        return False


def run_tests():
    """Run pytest suite."""
    print("\nRunning tests...")
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
    )
    return result.returncode == 0


def validate_structure():
    """Validate project structure."""
    print("\nValidating project structure...")
    
    required_dirs = [
        "src",
        "src/data",
        "src/models",
        "src/train",
        "api",
        "tests",
        "data/raw",
        "data/processed",
        ".github/workflows",
    ]
    
    required_files = [
        "pyproject.toml",
        "README.md",
        "Dockerfile",
        ".gitignore",
        "src/__init__.py",
        "tests/__init__.py",
    ]
    
    for d in required_dirs:
        path = Path(d)
        if path.is_dir():
            print(f"✓ {d}/")
        else:
            print(f"✗ {d}/ missing")
            return False
    
    for f in required_files:
        path = Path(f)
        if path.is_file():
            print(f"✓ {f}")
        else:
            print(f"✗ {f} missing")
            return False
    
    return True


def main():
    """Run all setup and validation checks."""
    print("="*80)
    print("ECG ARRHYTHMIA DETECTION - PROJECT SETUP & VALIDATION")
    print("="*80)
    
    print("\n1. Checking Python environment...")
    check_python_version()
    
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("\nTo install dependencies, run:")
        print("  pip install -e '.[dev]'")
        sys.exit(1)
    
    print("\n3. Validating project structure...")
    if not validate_structure():
        print("Project structure validation failed")
        sys.exit(1)
    
    print("\n4. Generating synthetic test data...")
    if not generate_synthetic_data():
        print("Failed to generate synthetic data")
        sys.exit(1)
    
    print("\n5. Running test suite...")
    if not run_tests():
        print("Some tests failed. Review output above.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("="*80)
    print("\nProject is ready for Stage 1 training:")
    print("  python -m src.train.train_baseline")
    print("\nOr view API documentation:")
    print("  uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
