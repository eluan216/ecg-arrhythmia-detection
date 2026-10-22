# ECG Arrhythmia Detection

Inter-patient ECG beat classification on **MIT-BIH**, with a multi-model benchmark, FastAPI serving, Docker, and CI.

I built this as an ML engineering portfolio piece: the evaluation protocol is the product, not just the classifier.

**⚠️ Research / education only. NOT a medical device. Do not use for clinical decisions.**

---

## Evaluation protocol (default)

Results below use an **inter-patient split**: no beat from the same patient appears in both train and test.

| Setting | Value |
|---------|--------|
| Dataset | MIT-BIH Arrhythmia Database |
| Classes | AAMI EC57: N, V, S, F, Q |
| Excluded records | 102, 104, 107 (paced — common literature exclusion) |
| Split | Patient-level, `train_ratio=0.8`, `seed=42` |
| Train / test patients | Written to `models/patient_split.json` |
| Features | 15-D morphology (see [`src/features.md`](src/features.md)) |
| CV | GroupKFold over **training patients** (mean ± std macro-F1) |
| Selection | Highest **macro-F1** on inter-patient holdout |

After you run the real pipeline, the numbers below should be replaced by the JSON in `models/inter_patient_results.json`.

### Results (MIT-BIH, inter-patient)

> Run once with data on disk, then paste measured values here.

```bash
pip install -e .
python -m src.data.download
python -m src.eval.inter_patient
```

| Model | Macro-F1 | Weighted-F1 | Accuracy | GroupKFold macro-F1 |
|-------|----------|-------------|----------|---------------------|
| Logistic Regression | *run pipeline* | *…* | *…* | mean ± std |
| Random Forest | *run pipeline* | *…* | *…* | mean ± std |

**Per-class F1 (best model):** N · V · S · F · Q — from `inter_patient_results.json`.

**Top confusions:** also in that JSON (`top_confusions`). Typical hard pairs under inter-patient evaluation: S↔N and rare F/Q support.

Synthetic offline benchmark (not inter-patient) remains available for CI and quick checks:

```bash
python -m src.pipeline.benchmark
```

---

## Quick start

### Offline multi-model table (synthetic)

```bash
git clone https://github.com/eluan216/ecg-arrhythmia-detection.git
cd ecg-arrhythmia-detection
pip install -e ".[dev]"
python -m src.pipeline.benchmark
pytest tests/ -q
```

### Real MIT-BIH inter-patient path

```bash
pip install -e .
python -m src.data.download
python -m src.eval.inter_patient          # RF + LR, GroupKFold, split file
python -m src.train.train_cnn            # 1D-CNN (uses processed/synthetic if no NPZ)
python -m src.evaluate                   # comparison table from metrics JSON
uvicorn api.main:app --reload
```

Docker:

```bash
docker build -t ecg-arrhythmia-api:latest .
docker run -p 8000:8000 ecg-arrhythmia-api:latest
```

---

## Architecture

```text
MIT-BIH records
      │
      ▼
R-peak windows → AAMI map (N/V/S/F/Q)
      │
      ├── morphology features (15-D) → RF / Logistic Regression
      └── raw windows               → 1D-CNN (Stage 2)
      │
Inter-patient holdout + GroupKFold
      │
metrics JSON · confusion matrix · top confusions
      │
FastAPI → Docker → CI
```

---

## Design decisions

- **Macro-F1 over accuracy** — minority arrhythmia classes must not be hidden under N-majority accuracy  
- **Inter-patient default** — beat-level random splits leak patient identity  
- **GroupKFold on train patients** — stability without mixing test patients into CV  
- **Documented exclusions** — paced records 102/104/107  
- **Synthetic path** — CI and first-run experience without PhysioNet  

---

## Error analysis & limits

- Rare classes (F, Q) have low support; per-class F1 is the honest view  
- Inter-patient generalization is harder than random beat splits — treat high synthetic F1 with skepticism  
- Morphology features are not full P-QRS-T delineation  
- Noise, electrode placement, and domain shift to other devices are out of scope  

---

## Repository layout

```text
src/data/        download, segment, synthetic
src/eval/        inter-patient protocol (default rigor path)
src/models/      RF baseline, 1D-CNN
src/train/       training scripts
src/pipeline/    offline multi-model benchmark
src/features.md  morphology feature definitions
api/             FastAPI
tests/
```

---

## Reproducibility

| Artifact | Path |
|----------|------|
| Patient split | `models/patient_split.json` |
| Full metrics | `models/inter_patient_results.json` |
| Seed | `42` (see eval module) |
| Package version | `pyproject.toml` |

---

## Author

**Oguma Eluanatein Odo**  
[GitHub](https://github.com/eluan216) · [LinkedIn](https://linkedin.com/in/eluanatein-oguma-5552571b6)

---

## License

MIT

---

## Disclaimer

**Portfolio / research / education only.** Not a medical device. Not for clinical use.
