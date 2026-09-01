# ECG Arrhythmia Detection & Deployment Pipeline

A portfolio project demonstrating end-to-end ML engineering discipline: data handling, model evaluation rigor, testing, CI/CD, and containerized deployment.

**⚠️ DISCLAIMER:** This is a **research and educational project only** and is **NOT** a medical diagnostic device. It must not be used, presented, or marketed as a clinical tool. Use for portfolio demonstration and learning purposes only.

---

## Project Overview

This project builds a complete pipeline to classify individual ECG heartbeats into five clinically-relevant arrhythmia categories using the **MIT-BIH Arrhythmia Database** (a field-standard benchmark dataset). The pipeline includes:

- **Data Pipeline**: Download, parse, and segment MIT-BIH records into R-peak-centered beat windows
- **Baseline Model**: scikit-learn Random Forest on handcrafted morphology features
- **Deep Learning Model**: PyTorch 1D-CNN on raw segmented beats
- **Rigorous Evaluation**: Per-class metrics (precision/recall/F1), confusion matrices, and inter-patient train/test splits
- **Experiment Tracking**: MLflow for metrics and model artifacts
- **API**: FastAPI service for inference
- **Containerization**: Docker image for reproducible deployment
- **CI/CD**: GitHub Actions for lint, test, and automated build on merge
- **Deployment**: Live demo on Render free tier

---

## Quick Start

### Prerequisites

- Python 3.9+
- Git
- (Optional) Docker for containerized deployment

### Setup

1. **Clone the repository:**

```bash
git clone <repository-url>
cd ecg-arrhythmia-detection
```

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. **Install dependencies:**

```bash
pip install -e ".[dev]"
```

1. **Download and process the dataset:**

```bash
python -m src.data.download
```

### Stage 1: Baseline Model (scikit-learn)

```bash
python -m src.train.train_baseline
```

Trains a Random Forest on handcrafted ECG features and logs results to MLflow.

### Stage 2: Deep Learning Model (PyTorch CNN)

```bash
python -m src.train.train_cnn
```

Trains a 1D-CNN on raw beat windows, tracks experiments via MLflow.

### Run Tests

```bash
pytest tests/ -v
```

### Evaluate Models

```bash
python -m src.evaluate
```

Generates comparison table and confusion matrices for both models.

### Start API Server (Local)

```bash
uvicorn api.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Dataset: MIT-BIH Arrhythmia Database

- **48 patient records**, ~30 minutes each, ~110,000 individual beat annotations
- **Sampling rate**: 360 Hz
- **Resolution**: 11-bit over 10 mV range
- **Lead configuration**: 2-channel (typically MLII + V1)
- **Annotations**: Beat-level labels independently reviewed by multiple cardiologists
- **Source**: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

### AAMI EC57 Class Mapping

Raw MIT-BIH annotation symbols are mapped to five standardized AAMI classes:

| Class | Code | Description | Examples |
| --- | --- | --- | --- |
| Normal | N | Normal beat | N, L, R |
| Ventricular Ectopic | V | Ventricular premature beat | V, \[, \] |
| Supraventricular Ectopic | S | Atrial/supraventricular premature beat | A, a, J, S |
| Fusion | F | Fusion of normal and ventricular | F |
| Unknown/Paced | Q | Unknown or paced beats | /, Q, e, \| |

**Note on class imbalance:** The dataset is heavily imbalanced. In a typical split, class N dominates (~95%), while classes S and F may have <200 examples. This motivates the evaluation strategy in Section 6 of the PRD.

---

## Architecture

### Data Flow

```text
PhysioNet/MIT-BIH Records
    ↓
Preprocessing (wfdb, R-peak alignment)
    ↓
Feature/Window Store (NPZ/Parquet)
    ↓
┌───────────────────────────────────────┐
│   Train/Test Split (inter-patient)    │
└───────────────────────────────────────┘
    ↓
    ├─→ scikit-learn Baseline (Random Forest on features)
    │
    └─→ PyTorch 1D-CNN (raw beat windows)
    
    Both ↓
    
    MLflow Tracking → Evaluation Metrics → README Comparison Table
    
    Best Model ↓
    
    FastAPI Service ↓
    
    Docker → Render Deployment
```

### Repository Structure

```text
ecg-arrhythmia-detection/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint + pytest on every push/PR
│       └── docker-build.yml    # Build & push on merge to main
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # Download MIT-BIH from PhysioNet
│   │   ├── segment.py          # R-peak windowing + AAMI mapping
│   │   └── dataset.py          # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_sklearn.py # Random Forest model
│   │   └── cnn_pytorch.py      # 1D-CNN model
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_baseline.py   # Train scikit-learn baseline
│   │   └── train_cnn.py        # Train PyTorch CNN
│   └── evaluate.py             # Generate comparison metrics
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   └── schemas.py              # Request/response models
├── tests/
│   ├── __init__.py
│   ├── test_segmentation.py
│   ├── test_dataset.py
│   └── test_api.py
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis (research only)
├── data/
│   ├── raw/                    # MIT-BIH records (gitignored)
│   └── processed/              # Segmented beats (gitignored)
├── Dockerfile
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Model Evaluation

Both models are evaluated on an **inter-patient train/test split** (80% of patients for training, 20% for testing) to ensure generalization and avoid inflated accuracy from intra-patient data leakage.

### Metrics Reported

For each class and macro-averaged:

- **Precision**: TP / (TP + FP)
- **Recall (Sensitivity)**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Raw counts and normalized percentages

### Why Not Just Accuracy?

Class imbalance means accuracy alone is misleading. For example:

- A naive classifier that always predicts "Normal" would achieve ~95% accuracy but have 0% recall on rare classes (V, S, F).
- Per-class metrics expose whether the model is actually learning the minority classes or just defaulting to the majority class.

The README comparison table prioritizes **precision/recall/F1 per class** over overall accuracy.

---

## Deployment

### Local Development

Start the API server:

```bash
uvicorn api.main:app --reload
```

### Docker

Build and run the image locally:

```bash
docker build -t ecg-arrhythmia-api:latest .
docker run -p 8000:8000 ecg-arrhythmia-api:latest
```

### Live Demo

The model is deployed on **Render** free tier. Visit the deployed API to test the model:

- **Note on cold starts**: The free tier spins down after 15 minutes of inactivity. The first request after inactivity will take ~30–60 seconds to respond as the service warms up. Subsequent requests are immediate. This is an infrastructure constraint, not a model bug.

**Deployed endpoint:** [To be added after deployment]

---

## Development Workflow

### Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Coverage report is generated in `htmlcov/index.html`.

### Code Quality

Linting and formatting are enforced in CI:

```bash
black src/ api/ tests/
ruff check src/ api/ tests/
mypy src/ api/
```

### Experiment Tracking

MLflow tracks metrics and models locally. View the UI:

```bash
mlflow ui
```

Visit `http://localhost:5000` to compare experiments.

---

## Milestones

| Stage | Deliverable | Status |
| --- | --- | --- |
| 1 | Data pipeline + sklearn baseline + pytest + CI | In Progress |
| 2 | PyTorch 1D-CNN + MLflow + evaluation report + CD | Planned |
| 3 | FastAPI + Docker + Render deployment + final README | Planned |

Each stage produces an independently presentable portfolio artifact.

---

## Known Constraints & Design Decisions

### CPU-Only Training

- The project is designed to run entirely on CPU (no GPU required). CNNs are shallow (2–4 conv layers) and beat windows are short (~250–360 samples) to keep training time reasonable.
- Honest about tradeoffs: If training takes 10 minutes on CPU, that's documented in the README, not hidden.

### Render Free-Tier Cold Starts

- Render's free tier doesn't scale to zero instantly like paid tiers. A request after 15+ minutes of inactivity will take ~30–60s to respond.
- This is noted in the API documentation so users understand it's an infrastructure choice, not a model performance issue.

### Class Imbalance Strategy

- Rather than upsampling/downsampling, the evaluation strategy compensates: per-class metrics and macro-averaged F1 ensure minority classes are not hidden by overall accuracy.
- Loss weighting may be explored in Stage 2 if needed.

### Inter-Patient Splits

- Train/test splits are stratified by patient to avoid data leakage and ensure the model generalizes to unseen patients.
- This is more rigorous than beat-level splits and is the standard in published benchmarks.

---

## Future Work (Out of Scope for v1)

- Real-time/streaming ECG ingestion
- 12-lead ECG support (currently 2-lead only)
- Automated retraining pipeline (CD triggered by new data)
- Attention mechanisms or transformer architectures
- Uncertainty quantification (prediction confidence calibration)
- Mobile or wearable integration

---

## References

- PhysioNet: MIT-BIH Arrhythmia Database
- "A Hybrid Deep CNN Model for Abnormal Arrhythmia Detection" (PMC)
- "A novel hybrid CNN-transformer model for arrhythmia detection" (Nature Scientific Reports)
- [lxdv/ecg-classification](https://github.com/lxdv/ecg-classification) — PyTorch implementation precedent
- AAMI EC57 standard and class mapping references

---

## License

MIT License. See LICENSE file for details.

---

## Disclaimer (Repeated for Emphasis)

**This project is for portfolio, research, and educational purposes only.** It is not approved, validated, or intended for use as a medical device or diagnostic tool. Do not rely on this system for any clinical decision-making. Always consult a qualified healthcare professional for medical advice.
