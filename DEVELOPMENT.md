# Development Guide

This guide covers setting up the ECG Arrhythmia Detection project for development and testing.

## Prerequisites

- **Python 3.9+** (required)
- **pip** (Python package manager)
- **Git** (for version control)
- Optional: **Docker** (for containerized development)

## Quick Start (Windows)

1. **Run the setup script:**

```cmd
setup.bat
```

This script will:

- Create a Python virtual environment
- Install all dependencies
- Activate the environment

1. **Run project validation:**

```cmd
python setup_project.py
```

This validates the project structure and generates synthetic test data.

## Quick Start (macOS/Linux)

1. **Create a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install dependencies:**

```bash
pip install -e ".[dev]"
```

1. **Run project validation:**

```bash
python setup_project.py
```

## Directory Structure

```text
ecg-arrhythmia-detection/
├── src/                          # Main source code
│   ├── data/                     # Data loading and preprocessing
│   │   ├── download.py           # Download MIT-BIH from PhysioNet
│   │   ├── segment.py            # Beat segmentation and features
│   │   ├── dataset.py            # PyTorch Dataset class
│   │   └── synthetic.py          # Synthetic data generation
│   ├── models/                   # Model definitions
│   │   ├── baseline_sklearn.py   # Random Forest model
│   │   └── cnn_pytorch.py        # 1D-CNN model
│   ├── train/                    # Training scripts
│   │   ├── train_baseline.py     # Train scikit-learn baseline
│   │   └── train_cnn.py          # Train PyTorch CNN
│   └── evaluate.py               # Model evaluation and comparison
├── api/                          # FastAPI inference service
├── tests/                        # Unit and integration tests
├── data/
│   ├── raw/                      # Raw MIT-BIH records (gitignored)
│   └── processed/                # Processed data (gitignored)
├── notebooks/                    # Jupyter notebooks for exploration
├── setup_project.py              # Setup validation script
└── setup.bat                     # Windows setup script
```

## Common Development Tasks

### Generate Synthetic Test Data

The project includes synthetic data generation for testing without requiring the full MIT-BIH dataset:

```python
python -c "from src.data.synthetic import save_synthetic_dataset; save_synthetic_dataset()"
```

This generates `data/processed/synthetic_beats.npz` with 1000 balanced beats (200 per class).

### Run Tests

Run all tests with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Coverage report is generated in `htmlcov/index.html`.

Run specific test file:

```bash
pytest tests/test_segmentation.py -v
```

Run with specific markers:

```bash
pytest -m "not slow" -v
```

### Code Quality

Format code with black:

```bash
black src/ api/ tests/
```

Check for linting issues:

```bash
ruff check src/ api/ tests/
```

Check type hints:

```bash
mypy src/ api/
```

### Train Models

Generate synthetic data first:

```bash
python -c "from src.data.synthetic import save_synthetic_dataset; save_synthetic_dataset()"
```

Train the scikit-learn baseline:

```bash
python -m src.train.train_baseline
```

This will:

1. Load synthetic data
2. Train a Random Forest model
3. Evaluate on test set
4. Log metrics to MLflow
5. Save results to `models/baseline_metrics.json`

### Start API Server

```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### View Experiment Tracking

Start MLflow UI:

```bash
mlflow ui
```

Visit `http://localhost:5000` to view experiment metrics.

### Download Real Data

To download the actual MIT-BIH dataset (requires internet connection):

```bash
python -c "from src.data.download import download_mitbih; download_mitbih()"
```

This downloads ~100MB of ECG records from PhysioNet. Records are cached in `data/raw/`.

## Git Workflow

### Initialize Repository

```bash
git init
git add .
git commit -m "Initial project setup"
```

### Before Pushing

1. **Run tests:**

```bash
pytest tests/ -v
```

1. **Check code quality:**

```bash
black src/ api/ tests/
ruff check src/ api/ tests/
```

1. **Commit:**

```bash
git add .
git commit -m "Description of changes"
```

## Troubleshooting

### ImportError: No module named 'torch'

Install PyTorch:

```bash
pip install torch torchvision
```

### ImportError: No module named 'wfdb'

Install wfdb for MIT-BIH data loading:

```bash
pip install wfdb
```

### Virtual environment not activated

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Tests fail with ImportError

Ensure the package is installed in development mode:

```bash
pip install -e ".[dev]"
```

## Project Stages

### Stage 1 (Current)

- ✓ Data pipeline (download, segment, features)
- ✓ Scikit-learn baseline (Random Forest)
- ✓ Test suite
- ✓ GitHub Actions CI/CD workflow
- [ ] Push to GitHub and verify CI

### Stage 2 (In Progress)

- [ ] PyTorch 1D-CNN model
- [ ] MLflow experiment tracking
- [ ] Model comparison report
- [ ] CD pipeline (build Docker on merge)

### Stage 3 (Future)

- [ ] FastAPI deployment
- [ ] Docker containerization
- [ ] Render deployment
- [ ] Final documentation

## Contributing

See `CONTRIBUTING.md` for guidelines on:

- Code style
- Testing requirements
- Pull request process
- Issue reporting

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

For issues or questions:

1. Check the README.md for general information
2. Review CONTRIBUTING.md for development practices
3. Open an issue on GitHub
