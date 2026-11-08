# Reproduce the research experiments

I separate three questions: whether a model generalizes to unseen patients,
whether rhythm context helps, and how frozen models behave under noise or a new dataset.

## Environment

Use Python 3.11 in a new virtual environment:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-research.txt -e ".[dev]"
```

MLflow is optional: install `.[tracking]` if you use the historical tracking helper.
The supported research suite writes plain JSON and does not need a tracking server.

## Patient-separated comparison and feature ablation

```bash
python -m src.eval.research_suite --download --epochs 10 --bootstrap 200
```

This is a complete real-data run, with no beat cap or synthetic fallback. Ten is
the predeclared maximum epoch budget, not a parameter chosen after viewing test results.
CNN selection uses inner validation subjects; the selected epoch count is refit
on all outer training subjects. RF morphology and RF morphology-plus-RR use fixed
200-tree parameters. All three models share the outer holdout.

RR features are previous RR in seconds, previous RR divided by the mean of up to
ten completed intervals, and that historical mean. The first beat has zeros.
Intervals use accepted reference beats, so missed detections in a real system
would introduce an additional source of error. No next-beat interval is used.

Outputs in `artifacts/research/`:

| File | Contents |
| --- | --- |
| `comparison.json` | Per-class metrics, confusion matrices, uncertainty, timings and protocol |
| `RESULTS.md` | Compact comparison table |
| `baseline_model.pkl` | Morphology RF compatible with the API |
| `rhythm_model.pkl` | Offline morphology-plus-RR ablation; do not substitute into the API |
| `cnn_model.pth` | CNN refit after validation-only epoch selection |
| `explorer_samples.json` | Curated held-out examples with reference and model outputs |
| `manifest.json` | File hashes, feature contract, classes and environment |
| `requirements-lock.txt` | Exact core scientific package versions |

The RF training time includes feature extraction, validation fitting and final
refitting. CNN time includes validation selection and refitting. Inference timing
is batch throughput including preprocessing, not single-request API latency.
Use the same hardware/environment for meaningful timing comparisons.

Completed bundles are not overwritten. Choose `--output-dir` for another run.
Change the configuration only using training/validation evidence. Report that
the historical holdout is a development benchmark once it has informed design.

## Noise robustness

```bash
python -m src.eval.external --dataset nstdb --download --data-dir data/nstdb --output artifacts/research/nstdb.json
```

The runner checks the bundle's training records, excludes overlapping clean
source patients, and scores annotation centers in noise-on segments only.
NSTDB adds electrode-motion noise after five minutes, in alternating two-minute
segments at SNRs 24, 18, 12, 6, 0 and -6 dB. This protocol has very few source patients;
no population-level confidence interval is claimed.

## External dataset

```bash
python -m src.eval.external --dataset incartdb --download --data-dir data/incart --output artifacts/research/incart.json
```

This downloads all 75 INCART records (roughly 0.8 GB), selects lead II in mV,
resamples whole signals from 257 to 360 Hz before segmentation, rounds the
reference sample locations, and evaluates frozen morphology RF/CNN weights.
It does not select models or fit anything to external labels. Repeated records
share patients; the report gives pooled metrics without pretending records are
independent patients. Label availability and reference alignment differ from MIT-BIH.
Mapped annotations outside the signal are excluded explicitly and recorded with
their original sample locations in the report's per-record annotation audit.
The default MIT-BIH loader remains strict about boundaries and sampling rate.

## Figures and explorer

```bash
python -m src.eval.plots
# macOS/Linux:
ECG_MODEL_DIR=artifacts/research uvicorn api.main:app --reload
```

On PowerShell, set `$env:ECG_MODEL_DIR = "artifacts/research"` first.
Open `http://localhost:8000/explorer`. Without a bundle, the explorer uses clearly
labeled synthetic fixtures and does not invent model predictions.

## Automated execution

The research workflow runs the comparison, NSTDB, INCART and figure generation.
It uploads the bundle and figures for 30 days. Download a successful artifact
for local exploration; use the source commit and report provenance to identify it.
The container workflow separately builds the image and checks readiness, inference
and the explorer with an explicitly synthetic smoke-test model.

## Sources

- [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) and [lead details](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm).
- [NSTDB noise construction and timing](https://physionet.org/content/nstdb/1.0.0/).
- [INCART sampling, leads and cohort](https://physionet.org/content/incartdb/1.0.0/).
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
- [SciPy polyphase resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html).
