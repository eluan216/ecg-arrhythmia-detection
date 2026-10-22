# Status

**Active hardening — inter-patient rigor path**

| Area | State |
|------|--------|
| Inter-patient split + patient ID file | Done (`src/eval/inter_patient.py`) |
| GroupKFold on train patients | Done |
| Morphology feature doc | Done (`src/features.md`) |
| Offline synthetic benchmark | Done |
| MIT-BIH measured results in README | Run `python -m src.eval.inter_patient` then publish numbers |
| CNN inter-patient parity | Partial (train_cnn still prefers synthetic/processed) |
| Error analysis section | Structured via `top_confusions` in results JSON |
| Doc consolidation | STAGE_* reports superseded by README + DEVELOPMENT.md |

Release target: portfolio-v1 after measured MIT-BIH table is committed.

eluan216 — 2026-10-22
