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

## Open / next

- **1D-CNN on the same subject-disjoint partition as RF** — architecture and training entrypoint exist; inter-patient MIT-BIH numbers are **not** in the results table yet. Train only after exporting beats with the same seed-42 patient split; do not mix random-beat CNN metrics with the published RF table.
- Optional: commit a small trusted baseline artifact path for API demos if you do not want to retrain locally.

## Offline demos

```bash
python quickstart.py                    # explicitly synthetic
python -m src.train.train_baseline --allow-synthetic
python -m src.train.train_cnn --allow-synthetic
python -m src.eval.inter_patient --allow-synthetic
```

Published protocol path:

```bash
python -m src.eval.full_mitbih
```

eluan216 — 2026-11-05
