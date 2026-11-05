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
| Excluded records | 102, 104, 107 (repository protocol; 217 retained) |
| Split | Patient-level, `train_ratio=0.8`, `seed=42` |
| Train / test patients | Written to `artifacts/eval/patient_split.json` |
| Features | 15-D morphology (see [`src/features.md`](src/features.md)) |
| CV | GroupKFold over **training patients** (mean ± std macro-F1) |
| Selection | Fixed Random Forest; holdout reserved for final reporting |

Measured results below were produced by the full real-data run on 2026-11-05, with no beat cap or synthetic fallback.

### Results (MIT-BIH, inter-patient)

[Full evaluation run](https://github.com/eluan216/ecg-arrhythmia-detection/actions/runs/42090136918)
· [Measured report](reports/mitbih/2026-11-05/full_report.json)
· [Patient split](reports/mitbih/2026-11-05/patient_split.json).

All **45 included records** were loaded: **102,941 accepted beats** in total.
Training uses 82,007 beats from 35 subjects (36 records); the held-out test set
uses 20,934 beats from 9 different subjects (9 records). Records 201 and 202
remain together in training. This is the repository's custom seed-42 split,
not the fixed DS1/DS2 split used in some publications.

The dataset has 48 records; this protocol excludes 102, 104, and 107 and
retains 217. These results therefore describe the full **45-record protocol**,
not an evaluation of all 48 records. Each input is a 250-sample window centered
on a reference annotation, using channel 0 at 360 Hz. Beat localization is
provided by the dataset annotations; this does not measure R-peak detection.

| Model | Macro-F1 | Weighted-F1 | Accuracy | 5-fold GroupKFold macro-F1 |
|-------|----------|-------------|----------|----------------------------|
| Random Forest (200 trees) | 35.83% | 94.96% | **94.79%** | 28.61% ± 4.02 percentage points |
| 1D-CNN (raw windows) | *not measured on this split* | — | — | — |

The model and seed were fixed before this run. Cross-validation uses training
subjects only; the held-out set was used for final reporting, not model selection.
Macro-F1 always averages all five declared classes, including zero-F1 classes.

| Class | Precision | Recall | F1 | Test beats |
|-------|-----------|--------|----|------------|
| N | 96.90% | 97.48% | 97.19% | 19,032 |
| V | 87.40% | 75.97% | 81.29% | 1,698 |
| S | 0.91% | 0.52% | 0.66% | 193 |
| F | 0.00% | 0.00% | 0.00% | 7 |
| Q | 0.00% | 0.00% | 0.00% | 4 |

**Interpretation:** high overall accuracy is dominated by normal beats
(19,032 of 20,934 test beats). The model performs poorly on supraventricular
beats and misses the rare F/Q test examples. It is not a reliable five-class
classifier on unseen patients. F/Q estimates are especially unstable because
their test supports are only 7 and 4.

Largest confusions: V→N (394), N→Q (189), S→N (188), N→V (182), and N→S (109).
The full report includes the complete confusion matrix in class order N/V/S/F/Q,
per-class metrics, file SHA-256 hashes, source commit, and package versions.

Reproduce the complete run:

```bash
pip install -e .
python -m src.eval.full_mitbih
```

The runner downloads data, checks that every included record is present and
evaluated, and writes metrics, split, provenance, and the trained model to
`artifacts/full-mitbih/`. The workflow also uploads these as the
`full-mitbih-results` artifact (30-day retention); the measured JSON report is
committed above for durable access. Core numerical package versions are saved
in the report. Source data:
[MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/),
Moody and Mark (2005), DOI: 10.13026/C2F305.

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

### API contract

Start with `uvicorn api.main:app --reload`. The API loads trusted local
`baseline_model.pkl` and/or `cnn_model.pth` artifacts. Set `ECG_MODEL_DIR`
to select a directory explicitly; the default search includes `models/`,
`artifacts/models/`, and `artifacts/eval/`.

POST `/predict` accepts `{"signal": [/* beat samples */], "sampling_rate": 360, "model": "auto"}`.
Supply 32–1000 finite samples at 360 Hz. `/health` reports liveness; `/ready` returns
503 until a model is loaded.

### Real MIT-BIH inter-patient path

```bash
pip install -e .
python -m src.data.download
python -m src.eval.inter_patient          # RF, GroupKFold, split file (fail closed)
# Optional offline smoke tests only:
# python -m src.train.train_baseline --allow-synthetic
# python -m src.train.train_cnn --allow-synthetic
uvicorn api.main:app --reload
```

**CNN note:** the 1D-CNN trainer is implemented, but inter-patient MIT-BIH CNN metrics are **not** published in the results table above. Do not mix synthetic/random-beat CNN scores with the measured RF protocol. To compare fairly, export beats on the same seed-42 patient split used by `src.eval.full_mitbih` / `reports/mitbih/2026-11-05/patient_split.json`, then train without `--allow-synthetic`.

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
      └── raw windows               → 1D-CNN (not yet measured inter-patient)
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
- **Documented exclusions** — repository protocol excludes 102/104/107; retains 217
- **Synthetic path is opt-in** — CI / `quickstart.py` / `--allow-synthetic` only; published metrics require real MIT-BIH

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
| Patient split | `artifacts/eval/patient_split.json` |
| Full metrics | `artifacts/eval/inter_patient_results.json` |
| Seed | `42` (see eval module) |
| Package version | `pyproject.toml` |
| Measured full-data report | `reports/mitbih/2026-11-05/full_report.json` |
| Full-data run outputs | `artifacts/full-mitbih/` |

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
