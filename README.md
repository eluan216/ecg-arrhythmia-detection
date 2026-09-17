# ECG Arrhythmia Detection & Deployment Pipeline

End-to-end ML engineering for ECG beat classification: multi-model benchmarking, stratified evaluation, MLflow tracking, FastAPI serving, and Docker.

**⚠️ DISCLAIMER:** Research and educational project only. **NOT** a medical diagnostic device. Do not use for clinical decisions.

---

## Engineering pattern

```text
Raw / synthetic beats
        ↓
Morphology features
        ↓
┌─────────────────────────────────┐
│ Model Benchmark                 │
│  Logistic Regression (baseline) │
│  Random Forest                  │
│  XGBoost / LightGBM (optional)  │
│  1D-CNN (raw windows, Stage 2)  │
└─────────────────────────────────┘
        ↓
Stratified 5-fold CV + holdout
        ↓
Metric comparison (Precision · Recall · F1 · ROC-AUC)
        ↓
Select model (macro-F1 under class imbalance)
        ↓
MLflow → FastAPI → Docker → CI
```

Candidate models are chosen for the problem (tabular morphology + optional 1D-CNN on windows)—not every algorithm for its own sake.

---

## Quick start (offline benchmark)

No PhysioNet download required for the multi-model table:

```bash
git clone https://github.com/eluan216/ecg-arrhythmia-detection.git
cd ecg-arrhythmia-detection
pip install -e ".[dev]"
# optional boosting models:
pip install -e ".[boost]"

python -m src.pipeline.benchmark
# or: python -m src.pipeline.benchmark --samples 300
```

This runs:

1. Synthetic AAMI-style beats
2. Morphology feature extraction
3. Logistic Regression · Random Forest · (XGBoost / LightGBM if installed)
4. Stratified 5-fold CV + holdout metrics
5. Model selection by **macro-F1**
6. Writes `models/benchmark_results.json`

Example output columns: Accuracy · Precision · Recall · F1 · ROC-AUC · CV-F1 ± std.

---

## Full MIT-BIH pipeline

```bash
pip install -e ".[dev]"
python -m src.data.download          # PhysioNet MIT-BIH
python -m src.train.train_baseline   # Random Forest on real features
python -m src.train.train_cnn        # 1D-CNN on beat windows
python -m src.evaluate               # Comparison table
uvicorn api.main:app --reload        # Inference API
```

### Docker

```bash
docker build -t ecg-arrhythmia-api:latest .
docker run -p 8000:8000 ecg-arrhythmia-api:latest
```

---

## Model benchmarking

Five AAMI classes (N, V, S, F, Q). Evaluation prioritizes **macro-F1** and per-class precision/recall because the problem is heavily imbalanced—accuracy alone is misleading.

| Model | Role |
| --- | --- |
| Logistic Regression | Linear baseline |
| Random Forest | Nonlinear ensemble benchmark |
| XGBoost / LightGBM | Gradient boosting (optional deps) |
| 1D-CNN | Deep model on raw windows (Stage 2) |

**Selection rule:** highest macro-F1 on holdout, with stratified CV as a stability check. Document the winner and why in experiment notes / MLflow.

Inter-patient splits are used on real MIT-BIH data to avoid leakage from beats of the same patient appearing in both train and test.

---

## Dataset: MIT-BIH Arrhythmia Database

- 48 records · ~110k annotated beats · 360 Hz  
- Source: [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)  
- Labels mapped to AAMI EC57 classes (N, V, S, F, Q)

---

## Repository structure

```text
ecg-arrhythmia-detection/
├── src/
│   ├── data/           # download, segment, synthetic
│   ├── models/         # RF baseline, 1D-CNN
│   ├── train/          # train scripts, stage2 compare
│   ├── pipeline/       # multi-model benchmark (this template)
│   └── evaluate.py
├── api/                # FastAPI
├── tests/
├── .github/workflows/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## API & deployment

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs`. Cold starts on free hosting may take 30–60s after idle periods.

---

## Design decisions

- **Macro-F1 over accuracy** — minority arrhythmia classes must not be hidden  
- **Stratified CV** — stable estimates under imbalance  
- **Inter-patient evaluation** on real data — no beat-level leakage  
- **CPU-friendly CNN** — shallow network, short windows  
- **Synthetic path** — anyone can run the benchmark without PhysioNet

---

## Author

**Oguma Eluanatein Odo**  
[GitHub](https://github.com/eluan216) · [LinkedIn](https://linkedin.com/in/eluanatein-oguma-5552571b6) · ogumaeluan@gmail.com

---

## License

MIT

---

## Disclaimer

**Portfolio / research / education only.** Not a medical device. Not for clinical use.
