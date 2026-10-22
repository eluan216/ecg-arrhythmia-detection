# Status

**Phase 1 (correctness) — implemented**

| Item | State |
|------|--------|
| Unified morphology features (`morphology_v1`) | Done — train + API share `src.data.segment.extract_morphology_features` |
| Feature parity tests | Done — `tests/test_feature_parity.py` |
| Annotation whitelist (unknown ≠ Q) | Done |
| Subject-disjoint split (201/202 together) | Done |
| ECG input contract (NaN/empty/rank/length) | Done |
| CORS not `*` by default | Done — `ECG_CORS_ORIGINS` |
| `/ready` vs `/health` | Done |
| Explicit dataset_type in metrics | Done for inter-patient path |
| Silent synthetic fallback | Still present in `train_baseline` / `train_cnn` for offline demos — **do not publish those metrics as MIT-BIH** |

**Next (Phase 2)**

- CNN on the same subject-disjoint partition as RF
- Require `--allow-synthetic` (fail closed when MIT-BIH missing)
- Measured MIT-BIH table committed to README
- Remove STAGE_* narrative docs after results freeze

eluan216 — 2026-10-23
