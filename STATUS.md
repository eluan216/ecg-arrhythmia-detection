# Status

**Frozen evaluation (2026-11-05):** inter-patient Random Forest on the full 45-record MIT-BIH protocol is measured and committed under `reports/mitbih/2026-11-05/`. See README for numbers, interpretation, and provenance.

## Correctness defaults

| Item | State |
|------|--------|
| Unified morphology features (`morphology_v1`) | Done — train + API share `src.data.segment.extract_morphology_features` |
| Feature parity tests | Done — `tests/test_feature_parity.py` |
| Annotation whitelist (unknown ≠ Q) | Done |
| Subject-disjoint split (201/202 together) | Done |
| ECG input contract (NaN/empty/rank/length) | Done |
| CORS not `*` by default | Done — `ECG_CORS_ORIGINS` |
| `/ready` vs `/health` | Done |
| Explicit `dataset_type` in metrics | Done for inter-patient path |
| Synthetic training fallback | **Fail closed** — `train_baseline` / `train_cnn` / `inter_patient` require `--allow-synthetic` (or `use_synthetic=True` from offline scripts). Never publish synthetic metrics as MIT-BIH. |
| Measured MIT-BIH RF table in README | Done |
| 1D-CNN inter-patient path | **Ready** — `python -m src.train.train_cnn --from-patient-split reports/mitbih/2026-11-05/patient_split.json` loads the same subject-disjoint partition, trains on raw 250-sample windows, and writes `models/cnn_metrics.json`. CNN numbers are **not** yet in the published results table (run after `python -m src.data.download`). |

## Open / next

- **Publish 1D-CNN inter-patient numbers** — run the `--from-patient-split` path (CPU ~30–60 min for 50 epochs; GPU faster), then fold the resulting macro-F1 / accuracy into the README results table and a `reports/mitbih/.../cnn_report.json` artifact. Do not mix synthetic/random-beat CNN scores with the measured RF protocol.
- Optional: commit a small trusted baseline artifact path for API demos if you do not want to retrain locally.

## Offline demos

```bash
python quickstart.py                    # explicitly synthetic
python -m src.train.train_baseline --allow-synthetic
python -m src.train.train_cnn --allow-synthetic
python -m src.eval.inter_patient --allow-synthetic
```

Published protocol path (RF):

```bash
python -m src.eval.full_mitbih
```

CNN on the same split (after download):

```bash
python -m src.data.download
python -m src.train.train_cnn \
  --from-patient-split reports/mitbih/2026-11-05/patient_split.json \
  --output-dir models/cnn_interpatient \
  --epochs 50
```

eluan216 — 2026-11-05
