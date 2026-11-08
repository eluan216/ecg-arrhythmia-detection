# Development

Use Python 3.11 and a virtual environment; see the [README](README.md) for platform-specific activation commands.

```bash
python -m pip install -e ".[dev]"
ruff check src/ api/ tests/
black --check --diff src/ api/ tests/
pytest tests/ -v --cov=src --cov-report=term-missing
```

Format modified Python files with Black before submitting changes. CI runs lint, formatting and tests on Python 3.9–3.11. Its mypy step is currently advisory.

## Data and training

`python -m src.pipeline.benchmark` is the offline synthetic demonstration.
`python -m src.eval.full_mitbih` is the complete measured Random Forest protocol.
The lower-level evaluator accepts `--data-dir`, `--output-dir`, `--seed` and
`--n-splits`. A beat cap changes the experiment and must be reported.

`--allow-synthetic` enables smoke-test fallback; do not label those results MIT-BIH.
CNN NPZ inputs do not prove subject independence; use `--from-patient-split`
for the real-data protocol. Choose epochs before viewing the test results.
Use training subjects alone for future validation or hyperparameter search.

Generated data, weights, caches and coverage outputs belong in ignored directories.
Preserve measured reports with the source commit, split and environment.
The older setup helpers and archived stage notes are historical conveniences;
the commands in the README are the supported starting point.

## Inference

Point `ECG_MODEL_DIR` at a trusted model directory and start
`uvicorn api.main:app --reload`. Restart the service after changing model files.
A healthy service can still be unready when no model loads; inspect `/ready`.
A pickle/joblib model must come from a trusted source and a compatible environment.

## Review expectations

Explain changes to data selection, lead policy, preprocessing, class order and
metrics. Add regression coverage for behavior changes. Keep synthetic results
separate from real-data evidence and do not overwrite historical reports with
new results. See [research notes](docs/research.md) for the current limitations.


## Research and explorer

The [experiment guide](docs/experiments.md) documents the fixed Python 3.11
scientific environment, validation-only model selection and external evaluation.
The explorer is served at `/explorer` by the same FastAPI process.
Its static HTML is included in Python packages and the Docker image.

MLflow is optional (`pip install -e ".[tracking]"`); it is not required by
the supported JSON-based experiment suite. Torchvision is not used.
Research bundles verify hashes and feature/environment compatibility before loading.
The RR model is an offline ablation and must not replace the API's morphology model.
